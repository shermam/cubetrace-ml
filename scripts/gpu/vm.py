"""The coordinator's driver for the GPU machine (see docs/GPU.md): creates a Compute Engine instance from
a Deep Learning VM image with scripts/gpu/startup.sh as its startup script and the run's settings as
metadata, follows its serial console, and deletes it. Application Default Credentials (the deploy service
account, with Compute Instance Admin and Service Account User) through the `gcs` extra's google-auth:

    uv run --no-sync python scripts/gpu/vm.py check                      # the project, the quotas, the SA
    uv run --no-sync python scripts/gpu/vm.py create --name ml-1 \
        --data-root gs://cubetrace-data/users/<uid> --out-prefix gs://cubetrace-data/features/2026-10-06 \
        [--encoders dinov2-vits14,resnet18] [--args "--split train --limit 20"] [--zone us-central1-a]
        [--machine g2-standard-8 | --machine n1-standard-8 --gpu nvidia-tesla-t4] [--spot] [--keep-up]
    uv run --no-sync python scripts/gpu/vm.py serial --name ml-1 [--follow]   # the console, to the status
    uv run --no-sync python scripts/gpu/vm.py status --name ml-1
    uv run --no-sync python scripts/gpu/vm.py delete --name ml-1

Nothing here prints a credential; the key file's path comes from GOOGLE_APPLICATION_CREDENTIALS.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import google.auth
from google.auth.transport.requests import AuthorizedSession

COMPUTE = "https://compute.googleapis.com/compute/v1"
IMAGE_FAMILY = (
    "projects/deeplearning-platform-release/global/images/family/common-cu129-ubuntu-2404-nvidia-580"
)
STATUS_LINE = re.compile(r"CUBETRACE-ML-STATUS (\d+)")
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def session() -> tuple[AuthorizedSession, str]:
    credentials, project = google.auth.default(scopes=SCOPES)
    if not project:
        raise SystemExit("no project in the credentials; set GOOGLE_CLOUD_PROJECT")
    return AuthorizedSession(credentials), project


def check(args: argparse.Namespace) -> int:
    s, project = session()
    print(f"project {project}")
    r = s.get(f"{COMPUTE}/projects/{project}", params={"fields": "defaultServiceAccount,quotas"}, timeout=30)
    if not r.ok:
        message = r.json().get("error", {}).get("message", r.text[:200])
        print(f"compute project read: {r.status_code}: {message}")
        print("  the deploy service account needs Compute Instance Admin (v1) on the project (docs/GPU.md)")
        return 1
    doc = r.json()
    print(f"default compute service account {doc.get('defaultServiceAccount')}")
    quotas = {q["metric"]: (q["limit"], q["usage"]) for q in doc.get("quotas", [])}
    for metric in ("GPUS_ALL_REGIONS",):
        print(f"  {metric}: {quotas.get(metric, 'not listed')}")
    r = s.get(f"{COMPUTE}/projects/{project}/regions/{args.region}", params={"fields": "quotas"}, timeout=30)
    if r.ok:
        quotas = {q["metric"]: (q["limit"], q["usage"]) for q in r.json().get("quotas", [])}
        for metric in sorted(quotas):
            if "GPU" in metric or metric in ("CPUS", "N2_CPUS", "PREEMPTIBLE_CPUS", "SSD_TOTAL_GB"):
                print(f"  {args.region} {metric}: limit {quotas[metric][0]:g}, usage {quotas[metric][1]:g}")
    r = s.get(f"{COMPUTE}/{IMAGE_FAMILY}", params={"fields": "name,creationTimestamp"}, timeout=30)
    print(f"image family: {r.json().get('name') if r.ok else r.status_code}")
    return 0


def instance_body(args: argparse.Namespace, project: str) -> dict:
    zone = args.zone
    startup = (Path(__file__).parent / "startup.sh").read_text()
    items = [
        {"key": "startup-script", "value": startup},
        {"key": "install-nvidia-driver", "value": "True"},
        {"key": "ml-data-root", "value": args.data_root},
        {"key": "ml-out-prefix", "value": args.out_prefix},
        {"key": "ml-encoders", "value": args.encoders},
        {"key": "ml-ref", "value": args.ref},
        {"key": "ml-shutdown", "value": "0" if args.keep_up else "1"},
    ]
    if args.args:
        items.append({"key": "ml-args", "value": args.args})
    if args.workers:
        items.append({"key": "ml-workers", "value": str(args.workers)})
    body: dict = {
        "name": args.name,
        "machineType": f"zones/{zone}/machineTypes/{args.machine}",
        "disks": [
            {
                "boot": True,
                "autoDelete": True,
                "initializeParams": {
                    "sourceImage": IMAGE_FAMILY,
                    "diskSizeGb": str(args.disk_gb),
                    "diskType": f"zones/{zone}/diskTypes/pd-balanced",
                },
            }
        ],
        "networkInterfaces": [
            {
                "network": "global/networks/default",
                "accessConfigs": [{"type": "ONE_TO_ONE_NAT", "name": "External NAT"}],
            }
        ],
        "serviceAccounts": [{"email": args.service_account or "default", "scopes": SCOPES}],
        "metadata": {"items": items},
        "scheduling": {"onHostMaintenance": "TERMINATE", "automaticRestart": False},
        "labels": {"cubetrace-ml": "features"},
    }
    if args.gpu:
        body["guestAccelerators"] = [
            {"acceleratorType": f"zones/{zone}/acceleratorTypes/{args.gpu}", "acceleratorCount": 1}
        ]
    if args.spot:
        body["scheduling"].update({"provisioningModel": "SPOT", "instanceTerminationAction": "DELETE"})
    return body


def wait_operation(s: AuthorizedSession, project: str, zone: str, op: dict, what: str) -> bool:
    name = op["name"]
    for _ in range(120):
        r = s.get(f"{COMPUTE}/projects/{project}/zones/{zone}/operations/{name}", timeout=30)
        doc = r.json()
        if doc.get("status") == "DONE":
            if doc.get("error"):
                for e in doc["error"].get("errors", []):
                    print(f"{what}: {e.get('code')}: {e.get('message')}")
                return False
            print(f"{what}: done")
            return True
        time.sleep(5)
    print(f"{what}: still running after 10 minutes")
    return False


def create(args: argparse.Namespace) -> int:
    if args.dry_run:
        body = instance_body(args, "<project>")
        shown = json.loads(json.dumps(body))
        shown["metadata"]["items"] = [
            {**i, "value": f"<{len(i['value'])} chars>"} if i["key"] == "startup-script" else i
            for i in shown["metadata"]["items"]
        ]
        print(json.dumps(shown, indent=2))
        return 0
    s, project = session()
    body = instance_body(args, project)
    r = s.post(f"{COMPUTE}/projects/{project}/zones/{args.zone}/instances", json=body, timeout=60)
    if not r.ok:
        err = r.json().get("error", {})
        print(f"create: {r.status_code}: {err.get('message', r.text[:300])}")
        return 1
    ok = wait_operation(s, project, args.zone, r.json(), f"create {args.name} in {args.zone}")
    return 0 if ok else 1


def serial_text(s: AuthorizedSession, project: str, zone: str, name: str, start: int = 0) -> tuple[str, int]:
    r = s.get(
        f"{COMPUTE}/projects/{project}/zones/{zone}/instances/{name}/serialPort",
        params={"port": 1, "start": start},
        timeout=60,
    )
    if not r.ok:
        return "", start
    doc = r.json()
    return doc.get("contents", ""), int(doc.get("next", start))


def status(args: argparse.Namespace) -> int:
    s, project = session()
    r = s.get(
        f"{COMPUTE}/projects/{project}/zones/{args.zone}/instances/{args.name}",
        params={"fields": "status,machineType,creationTimestamp,lastStartTimestamp,scheduling"},
        timeout=30,
    )
    if not r.ok:
        print(f"status: {r.status_code}: {r.json().get('error', {}).get('message', '')}")
        return 1
    doc = r.json()
    print(
        f"{args.name}: {doc.get('status')}, {doc.get('machineType', '').rsplit('/', 1)[-1]}, created "
        f"{doc.get('creationTimestamp')}, started {doc.get('lastStartTimestamp')}"
    )
    return 0


def serial(args: argparse.Namespace) -> int:
    """The console's lines from the startup script (and the status line); `--follow` polls until it
    appears, the machine stops, or `--minutes` pass."""
    s, project = session()
    start = 0
    shown = 0
    deadline = time.time() + 60 * args.minutes
    while True:
        text, start = serial_text(s, project, args.zone, args.name, start)
        lines = [ln for ln in text.splitlines() if "startup-script" in ln or "CUBETRACE-ML" in ln or args.all]
        for ln in lines:
            print(ln if args.all else ln.split("startup-script: ", 1)[-1])
        shown += len(lines)
        match = STATUS_LINE.search(text)
        if match:
            print(f"status line: {match.group(0)}")
            return int(match.group(1))
        if not args.follow or time.time() > deadline:
            return 0
        r = s.get(
            f"{COMPUTE}/projects/{project}/zones/{args.zone}/instances/{args.name}",
            params={"fields": "status"},
            timeout=30,
        )
        state = r.json().get("status") if r.ok else "?"
        if state in ("TERMINATED", "STOPPED", "STOPPING"):
            print(f"instance is {state}")
            return 0
        time.sleep(args.every)


def delete(args: argparse.Namespace) -> int:
    s, project = session()
    r = s.delete(f"{COMPUTE}/projects/{project}/zones/{args.zone}/instances/{args.name}", timeout=60)
    if not r.ok:
        print(f"delete: {r.status_code}: {r.json().get('error', {}).get('message', '')}")
        return 1
    return 0 if wait_operation(s, project, args.zone, r.json(), f"delete {args.name}") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--zone", default="us-central1-a")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("check", parents=[common])
    p.add_argument("--region", default="us-central1")
    p.set_defaults(run=check)
    p = commands.add_parser("create", parents=[common])
    p.add_argument("--name", required=True)
    p.add_argument("--data-root", required=True, help="gs://cubetrace-data/users/<uid>")
    p.add_argument("--out-prefix", required=True, help="gs://cubetrace-data/features/<run>")
    p.add_argument("--encoders", default="dinov2-vits14,resnet18")
    p.add_argument("--ref", default="main")
    p.add_argument("--args", default="", help="extra arguments of cubetrace-ml features")
    p.add_argument("--workers", type=int)
    p.add_argument("--machine", default="g2-standard-8", help="g2-standard-8 (an L4), or an n1 with --gpu")
    p.add_argument("--gpu", default="", help="an accelerator type for n1 machines, e.g. nvidia-tesla-t4")
    p.add_argument("--disk-gb", type=int, default=100)
    p.add_argument(
        "--service-account", default="", help="the machine's service account (default: the default one)"
    )
    p.add_argument("--spot", action="store_true", help="Spot pricing (the machine may stop at any time)")
    p.add_argument("--keep-up", action="store_true", help="do not shut the machine down at the end")
    p.add_argument("--dry-run", action="store_true", help="print the instance body and stop")
    p.set_defaults(run=create)
    p = commands.add_parser("status", parents=[common])
    p.add_argument("--name", required=True)
    p.set_defaults(run=status)
    p = commands.add_parser("serial", parents=[common])
    p.add_argument("--name", required=True)
    p.add_argument("--follow", action="store_true")
    p.add_argument("--all", action="store_true", help="every console line, not only the startup script's")
    p.add_argument("--every", type=int, default=60, help="seconds between polls with --follow")
    p.add_argument("--minutes", type=int, default=180, help="give up following after this long")
    p.set_defaults(run=serial)
    p = commands.add_parser("delete", parents=[common])
    p.add_argument("--name", required=True)
    p.set_defaults(run=delete)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
