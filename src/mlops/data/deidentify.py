"""Replaces patient identifiers before data leaves the raw zone."""

import hashlib
import hmac


def pseudonymize(patient_id: str, secret: str) -> str:
    """Returns a stable pseudonym for the patient.

    A keyed hash (HMAC): the same patient always gets the same pseudonym, so their images stay
    linked, but without the secret nobody can recompute it from a known patient ID.
    """
    if not secret:
        raise ValueError("The de-identification secret is empty")
    return hmac.new(secret.encode(), patient_id.encode(), hashlib.sha256).hexdigest()[:16]
