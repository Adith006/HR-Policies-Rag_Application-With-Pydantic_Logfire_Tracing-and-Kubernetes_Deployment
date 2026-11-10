
import os
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import HTTPException

# Set a test secret before importing the application.
TEST_SECRET = "test-only-authentication-secret-32-characters"
os.environ.setdefault("STREAMLIT_AUTH_SECRET", TEST_SECRET)

from app import main


ISSUER = "enterprise-hr-assistant"
AUDIENCE = "enterprise-hr-api"


@pytest.fixture(autouse=True)
def configure_test_secret(monkeypatch):
    monkeypatch.setattr(main, "STREAMLIT_AUTH_SECRET", TEST_SECRET)


def create_token(secret=TEST_SECRET, **overrides):
    now = datetime.now(timezone.utc)

    claims = {
        "sub": "test-user-123",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + timedelta(minutes=5),
        "email_verified": True,
    }

    claims.update(overrides)

    return jwt.encode(
        claims,
        secret,
        algorithm="HS256",
    )


def test_valid_token_returns_user_id():
    token = create_token()

    user_id = main.verify_google_token(
        authorization=f"Bearer {token}"
    )

    assert user_id == "test-user-123"


def test_missing_authorization_returns_401():
    with pytest.raises(HTTPException) as exc:
        main.verify_google_token(authorization=None)

    assert exc.value.status_code == 401


def test_invalid_authorization_scheme_returns_401():
    with pytest.raises(HTTPException) as exc:
        main.verify_google_token(
            authorization="Basic invalid-token"
        )

    assert exc.value.status_code == 401


def test_empty_bearer_token_returns_401():
    with pytest.raises(HTTPException) as exc:
        main.verify_google_token(authorization="Bearer ")

    assert exc.value.status_code == 401


def test_malformed_token_returns_401():
    with pytest.raises(HTTPException) as exc:
        main.verify_google_token(
            authorization="Bearer malformed.token.value"
        )

    assert exc.value.status_code == 401


def test_expired_token_returns_401():
    expired_token = create_token(
        exp=datetime.now(timezone.utc) - timedelta(minutes=5),
    )

    with pytest.raises(HTTPException) as exc:
        main.verify_google_token(
            authorization=f"Bearer {expired_token}"
        )

    assert exc.value.status_code == 401


@pytest.mark.parametrize(
    "claim",
    [
        {"iss": "incorrect-issuer"},
        {"aud": "incorrect-audience"},
    ],
)
def test_invalid_issuer_or_audience_returns_401(claim):
    token = create_token(**claim)

    with pytest.raises(HTTPException) as exc:
        main.verify_google_token(
            authorization=f"Bearer {token}"
        )

    assert exc.value.status_code == 401


def test_unverified_email_returns_401():
    token = create_token(email_verified=False)

    with pytest.raises(HTTPException) as exc:
        main.verify_google_token(
            authorization=f"Bearer {token}"
        )

    assert exc.value.status_code == 401


def test_missing_required_claim_returns_401():
    token = create_token()
    claims = jwt.decode(
        token,
        TEST_SECRET,
        algorithms=["HS256"],
        options={"verify_signature": False},
    )
    claims.pop("email_verified")

    token_without_claim = jwt.encode(
        claims,
        TEST_SECRET,
        algorithm="HS256",
    )

    with pytest.raises(HTTPException) as exc:
        main.verify_google_token(
            authorization=f"Bearer {token_without_claim}"
        )

    assert exc.value.status_code == 401


def test_empty_user_identity_returns_401():
    token = create_token(sub="")

    with pytest.raises(HTTPException) as exc:
        main.verify_google_token(
            authorization=f"Bearer {token}"
        )

    assert exc.value.status_code == 401


def test_wrong_signing_secret_returns_401():
    token = create_token(secret="another-test-secret")

    with pytest.raises(HTTPException) as exc:
        main.verify_google_token(
            authorization=f"Bearer {token}"
        )

    assert exc.value.status_code == 401


def test_missing_application_secret_returns_503(monkeypatch):
    monkeypatch.setattr(main, "STREAMLIT_AUTH_SECRET", None)

    with pytest.raises(HTTPException) as exc:
        main.verify_google_token(
            authorization="Bearer some-token"
        )

    assert exc.value.status_code == 503
