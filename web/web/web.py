import reflex as rx
import base64
import requests

from rxconfig import config


class State(rx.State):
    is_uploaded = False
    uploaded_files: list[rx.UploadFile] = []
    inference_host = config.inference_host
    is_inferring = False
    results: list[str]
    auth_token = config.auth_token

    @rx.event
    async def handle_upload(self, files: list[rx.UploadFile]):
        self.uploaded_files = []

        if not files:
            return

        for file in files:
            data = await file.read()
            path = rx.get_upload_dir() / file.name

            # write the file to server
            with path.open("wb") as f:
                f.write(data)

            self.uploaded_files.append(file.name)
        self.is_uploaded = True

    @rx.event
    async def run_inference(self):
        self.is_inferring = True
        base64_audios = []
        for filename in self.uploaded_files:
            file_path = rx.get_upload_dir() / filename
            with file_path.open("rb") as f:
                audio_bytes = f.read()
                audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                base64_audios.append(audio_base64)

        if not base64_audios:
            return

        response = requests.post(
            f"{self.inference_host}/predict",
            json={"audio_b64": base64_audios},
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.auth_token}"
            },
        )

        # Check if the request was successful
        if response.status_code != 200:
            print(f"Error from server: {response.status_code}")
            print(f"Raw response: {response.text}")
            self.is_inferring = False
            return

        try:
            self.results = response.json()["output"]
        except Exception as e:
            print(f"Failed to parse JSON. Raw response: {response.text}")

        self.is_inferring = False


def index() -> rx.Component:
    # Welcome Page (Index)
    return rx.container(
        rx.color_mode.button(position="top-right"),
        rx.vstack(
            rx.heading("Kentsel Ses (Urban Sound) Sınıflandırması", size="9"),
            rx.upload(
                rx.vstack(
                    rx.text(
                        "Sınıflandırılmak için sesinizi yükleyin. Maksimum 4 saniye, bundan fazlası kesilecektir.",
                    ),
                    rx.button(
                        "Upload Files",
                    ),
                    align="center"
                ),
                id="file_upload",
                border="2px dashed #ccc",
                padding="2em",
                multiple=True,
                accept={
                    "audio/*": [".wav", ".mp3", ".ogg"]
                },
                max_files=5,
                disabled=False,
                on_drop=State.handle_upload(rx.upload_files(upload_id="file_upload")),
            ),
            rx.cond(
                State.is_uploaded,
                # if true
                rx.vstack(
                    # Display uploaded files using rx.get_upload_url()
                    rx.foreach(
                        State.uploaded_files,
                        lambda filename: rx.audio(src=rx.get_upload_url(filename)),
                    ),
                    # run inference
                    rx.button(
                        "Run Inference",
                        on_click=State.run_inference,
                        # disable when no file uploaded, or while inferring
                        disabled=~State.is_uploaded | State.is_inferring
                    ),
                ),
                # else
                rx.text("No files uploaded yet.")
            ),
            rx.cond(
                State.results,
                # if true
                rx.foreach(
                    State.results,
                    lambda result: rx.text(result),
                ),
            ),
            spacing="5",
            justify="center",
            min_height="100vh",
        ),
    )


app = rx.App()
app.add_page(index)
