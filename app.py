from google_auth_oauthlib.flow import Flow

flow = Flow.from_client_secrets_file(
    "client_secret.json",
    scopes=["https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"],
    redirect_uri="http://127.0.0.1:5000/callback"  # Must match Google Cloud Console
)
