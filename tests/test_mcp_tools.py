from fastmcp import FastMCP
import paramiko
from pathlib import Path

mcp = FastMCP("ssh-tools")


@mcp.tool()
def ssh_exec_key(
    host: str,
    username: str,
    command: str,
    port: int = 22,
    key_path: str | None = None,
    timeout: int = 10,
) -> str:
    """
    Execute a command over SSH using public key authentication.
    """

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    if key_path:
        key_path = str(Path(key_path).expanduser())
        pkey = paramiko.RSAKey.from_private_key_file(key_path)
    else:
        # 使用 ~/.ssh/id_rsa / id_ed25519 / ssh-agent
        pkey = None

    client.connect(
        hostname=host,
        port=port,
        username=username,
        pkey=pkey,
        allow_agent=True,
        look_for_keys=True,
        timeout=timeout,
    )

    try:
        stdin, stdout, stderr = client.exec_command(command)
        out = stdout.read().decode(errors="ignore")
        err = stderr.read().decode(errors="ignore")
        return out + err
    finally:
        client.close()


if __name__ == "__main__":
    mcp.run()
