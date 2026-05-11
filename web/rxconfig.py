import reflex as rx
import os
from dotenv import load_dotenv

load_dotenv()

config = rx.Config(
    app_name="web",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ],
    inference_host="https://8000-dep-01krbdgf4c7hn0rck5r0nvprph-d.cloudspaces.litng.ai",
    auth_token=os.getenv("LIGHTNINGAI_TOKEN")
)
