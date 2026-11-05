"""A small in-memory stand-in for the docker SDK: just what the manager uses."""
from __future__ import annotations

import itertools
import subprocess
from types import SimpleNamespace

_ids = itertools.count(1)


def _match(labels, filters):
    for f in (filters or {}).get("label", []):
        k, _, v = f.partition("=")
        if k not in labels or (v and labels[k] != v):
            return False
    return True


class NotFound(Exception):
    pass


class FakeContainer:
    def __init__(self, client, name, labels, image="img"):
        self.client, self.name, self.labels, self.image = client, name, dict(labels), image
        self.id = f"{next(_ids):064x}"
        self.status = "running"
        self.attrs = {"State": {"Pid": 4242, "StartedAt": "x"}}
        self.removed = False
        self.exec_log = []

    def reload(self):
        pass

    def remove(self, force=False):
        self.removed = True
        self.client._containers.pop(self.name, None)

    def start(self):
        self.status = "running"

    def stop(self, timeout=10):
        self.status = "exited"

    def restart(self, timeout=10):
        self.status = "running"

    def kill(self):
        self.status = "exited"

    def pause(self):
        self.status = "paused"

    def unpause(self):
        self.status = "running"

    def exec_run(self, cmd, environment=None, **kw):
        self.exec_log.append(cmd)
        return SimpleNamespace(exit_code=0, output=b"")

    def logs(self, tail=200):
        return b"log line\n"

    def put_archive(self, path, data):
        self.client.archives.append((self.name, path, len(data)))
        return True


class FakeNetwork:
    def __init__(self, client, name, labels, subnet=""):
        self.client, self.name, self.labels = client, name, dict(labels)
        self.attrs = {"Labels": self.labels, "Containers": {}, "IPAM": {"Config": [{"Subnet": subnet}] if subnet else []}}
        self.connected = []

    def reload(self):
        pass

    def connect(self, c, **kw):
        self.connected.append((getattr(c, "name", c), kw))

    def disconnect(self, c, force=False):
        self.connected = [x for x in self.connected if x[0] != getattr(c, "name", c)]

    def remove(self):
        self.client._networks.pop(self.name, None)


class FakeVolume:
    def __init__(self, client, name, labels):
        self.client, self.name = client, name
        self.attrs = {"Labels": dict(labels)}

    def remove(self, force=False):
        self.client._volumes.pop(self.name, None)


class _Coll:
    def __init__(self, client, kind):
        self.c, self.kind = client, kind

    def _store(self):
        return getattr(self.c, f"_{self.kind}")

    def list(self, all=False, filters=None, names=None):
        out = []
        for o in self._store().values():
            labels = o.labels if self.kind != "volumes" else o.attrs["Labels"]
            if names and o.name not in names:
                continue
            if _match(labels, filters):
                out.append(o)
        return out

    def get(self, key):
        st = self._store()
        if key in st:
            return st[key]
        for o in st.values():
            if getattr(o, "id", None) == key:
                return o
        raise NotFound(key)

    def create(self, name, labels=None, ipam=None, **kw):
        if self.kind == "networks":
            subnet = ipam["Config"][0]["Subnet"] if ipam else ""
            o = FakeNetwork(self.c, name, labels or {}, subnet)
        else:
            o = FakeVolume(self.c, name, labels or {})
        self._store()[name] = o
        return o

    def run(self, image, name=None, labels=None, **kw):
        o = FakeContainer(self.c, name, labels or {}, image)
        self.c._containers[name] = o
        return o

    def prune(self, filters=None):
        return {"SpaceReclaimed": 0}


class FakeAPI:
    def __init__(self, client):
        self.c = client

    def create_host_config(self, **kw):
        return kw

    def create_networking_config(self, d):
        return d

    def create_endpoint_config(self, **kw):
        return kw

    def create_container(self, image, name=None, labels=None, **kw):
        o = FakeContainer(self.c, name, labels or {}, image)
        o.create_kwargs = kw
        self.c._containers[name] = o
        return {"Id": o.id}

    def start(self, cid):
        pass

    def prune_builds(self):
        return {"SpaceReclaimed": 0}


class FakeImages:
    def __init__(self, tags):
        self.tags = tags

    def list(self):
        return [SimpleNamespace(tags=self.tags)]

    def prune(self, filters=None):
        return {"SpaceReclaimed": 0}


class FakeDocker:
    def __init__(self):
        self._containers, self._networks, self._volumes = {}, {}, {}
        self.archives = []
        self.containers = _Coll(self, "containers")
        self.networks = _Coll(self, "networks")
        self.volumes = _Coll(self, "volumes")
        self.images = FakeImages(["ghost-lab/base:1", "ghost-lab/pg:18", "ghost-lab/web:1"])
        self.api = FakeAPI(self)

    def ping(self):
        return True


def runner_ok(fw_present=True, route_gw=None):
    """A subprocess.run stand-in: iptables -S returns the expected rules when fw_present."""
    from ghost_labs import net
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        out = ""
        if cmd[:1] == ["iptables"] and "-S" in cmd and fw_present:
            table = cmd[cmd.index("-t") + 1]
            out = "\n".join(r for t, r in net.expected_fw_rules("10.89.0.0/16", "10.88.0.0/24", "10.88.1.0/24") if t == table)
        if cmd[:1] == ["nsenter"] and "show" in cmd:
            out = f"default via {route_gw[0]} dev eth0" if route_gw and route_gw[0] else ""
        return subprocess.CompletedProcess(cmd, 0, out, "")
    run.calls = calls
    return run
