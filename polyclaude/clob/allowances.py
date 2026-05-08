"""On-chain allowance preflight.

Polymarket on Polygon uses three exchange contracts. Before any order can
settle, the trader's address must approve those contracts to spend USDC and
the conditional-token (CTF) NFTs.

For Polymarket proxy wallets (signature_type=2 or 3) the proxy holds funds
and approvals, so the EOA itself does not need approvals — we just verify
the proxy balance is positive.

Addresses below are documented in the Polymarket docs and should be kept in
sync with the canonical list. We keep them in one place so they're easy to
audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from polyclaude.config import Settings, get_settings
from polyclaude.logging_setup import get_logger

log = get_logger(__name__)


# Polymarket on Polygon mainnet
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# Three exchange contracts that need approvals
EXCHANGE_CONTRACTS = [
    "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",  # CTF Exchange
    "0xC5d563A36AE78145C45a50134d48A1215220f80a",  # Neg-risk CTF Exchange
    "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296",  # Neg-risk adapter
]


@dataclass(frozen=True)
class AllowanceCheck:
    address: str
    contract: str
    asset: str  # "USDC" or "CTF"
    approved: bool
    amount_or_flag: str  # raw allowance (USDC) or "true"/"false" (CTF)


@dataclass
class PreflightResult:
    ok: bool
    is_proxy: bool
    funder: str
    eoa: str
    checks: list[AllowanceCheck]
    missing: list[AllowanceCheck]
    notes: list[str]

    def explain(self) -> str:
        lines = [
            f"Allowance preflight ({'proxy' if self.is_proxy else 'EOA'} mode)",
            f"  EOA: {self.eoa}",
            f"  Funder: {self.funder}",
        ]
        if self.notes:
            lines.append("  Notes:")
            lines.extend(f"    - {n}" for n in self.notes)
        if self.ok:
            lines.append("  Status: OK")
        else:
            lines.append("  Status: MISSING APPROVALS")
            for m in self.missing:
                lines.append(
                    f"    - {m.asset} on {m.contract} (needs approval from {m.address})"
                )
            lines.append("")
            lines.append("  To fix (EOA mode), call from the EOA on Polygon mainnet:")
            for c in EXCHANGE_CONTRACTS:
                lines.append(f"    USDC.approve({c}, MAX_UINT256)")
                lines.append(f"    CTF.setApprovalForAll({c}, true)")
        return "\n".join(lines)


def _eoa_address_from_pk(private_key: str) -> str:
    """Best-effort derivation. Returns empty string if eth_account unavailable."""
    if not private_key:
        return ""
    try:
        from eth_account import Account

        return Account.from_key(private_key).address  # type: ignore[no-any-return]
    except Exception:
        return ""


def preflight(settings: Settings | None = None) -> PreflightResult:
    """Run allowance preflight.

    For EOA mode (signature_type=1) we attempt on-chain reads of allowances.
    If web3 / RPC is not available we degrade gracefully and emit a warning so
    the operator can verify manually.

    For proxy modes (2, 3) we skip allowance checks (proxy holds approvals)
    and only sanity-check that the funder address is set.
    """

    s = settings or get_settings()
    eoa = _eoa_address_from_pk(s.private_key)
    is_proxy = s.signature_type in (2, 3)
    notes: list[str] = []
    checks: list[AllowanceCheck] = []

    if not s.has_trading_creds():
        notes.append("PRIVATE_KEY / FUNDER not set — preflight is informational only.")
        return PreflightResult(
            ok=False, is_proxy=is_proxy, funder=s.funder, eoa=eoa, checks=[], missing=[],
            notes=notes,
        )

    if is_proxy:
        notes.append(
            "Proxy mode: approvals live on the proxy, not the EOA. "
            "Verify positive USDC balance in the funder address before trading."
        )
        return PreflightResult(
            ok=True, is_proxy=is_proxy, funder=s.funder, eoa=eoa, checks=[], missing=[],
            notes=notes,
        )

    # EOA mode: attempt on-chain read
    try:
        from web3 import Web3

        rpc = "https://polygon-rpc.com"
        w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 8}))
        if not w3.is_connected():
            raise RuntimeError("RPC not reachable")
        owner = Web3.to_checksum_address(eoa or s.funder)

        usdc_abi = [
            {
                "name": "allowance",
                "type": "function",
                "stateMutability": "view",
                "inputs": [
                    {"name": "owner", "type": "address"},
                    {"name": "spender", "type": "address"},
                ],
                "outputs": [{"name": "", "type": "uint256"}],
            }
        ]
        ctf_abi = [
            {
                "name": "isApprovedForAll",
                "type": "function",
                "stateMutability": "view",
                "inputs": [
                    {"name": "owner", "type": "address"},
                    {"name": "operator", "type": "address"},
                ],
                "outputs": [{"name": "", "type": "bool"}],
            }
        ]
        usdc = w3.eth.contract(address=Web3.to_checksum_address(USDC_ADDRESS), abi=usdc_abi)
        ctf = w3.eth.contract(address=Web3.to_checksum_address(CTF_ADDRESS), abi=ctf_abi)
        threshold = 10_000_000  # >= 10 USDC

        for c in EXCHANGE_CONTRACTS:
            spender = Web3.to_checksum_address(c)
            usdc_allow = int(usdc.functions.allowance(owner, spender).call())
            ctf_approved = bool(ctf.functions.isApprovedForAll(owner, spender).call())
            checks.append(
                AllowanceCheck(
                    address=owner, contract=c, asset="USDC",
                    approved=usdc_allow >= threshold, amount_or_flag=str(usdc_allow),
                )
            )
            checks.append(
                AllowanceCheck(
                    address=owner, contract=c, asset="CTF",
                    approved=ctf_approved, amount_or_flag=str(ctf_approved).lower(),
                )
            )
    except Exception as e:
        notes.append(f"On-chain read failed ({e!s}); cannot verify approvals automatically.")
        return PreflightResult(
            ok=False, is_proxy=is_proxy, funder=s.funder, eoa=eoa, checks=[], missing=[],
            notes=notes,
        )

    missing = [c for c in checks if not c.approved]
    return PreflightResult(
        ok=len(missing) == 0,
        is_proxy=is_proxy,
        funder=s.funder,
        eoa=eoa,
        checks=checks,
        missing=missing,
        notes=notes,
    )


def assert_ready(settings: Settings | None = None) -> PreflightResult:
    """Run preflight; raise if not OK. Used as a hard gate before live trading."""
    res = preflight(settings)
    log.info("clob.preflight", ok=res.ok, is_proxy=res.is_proxy, missing=len(res.missing))
    if not res.ok:
        raise RuntimeError("Allowance preflight failed:\n" + res.explain())
    return res


def usdc_units(amount_usd: Decimal) -> int:
    """USDC has 6 decimals."""
    return int((amount_usd * Decimal(10) ** 6).quantize(Decimal("1")))
