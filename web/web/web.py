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
        rx.color_mode.button(position="top-right", margin="1em"),
        rx.vstack(
            rx.heading("Kentsel Ses Sınıflandırması", size="8", text_align="center", margin_bottom="1em"),
            rx.upload(
                rx.vstack(
                    rx.icon("upload", color="gray", size=32),
                    rx.text(
                        "Sınıflandırılmak için ses dosyalarınızı sürükleyin veya tıklayarak seçin.",
                        text_align="center",
                        color="gray"
                    ),
                    rx.text(
                        "(Maksimum 4 saniye, daha uzun kayıtlar otomatik kesilecektir)",
                        size="2",
                        text_align="center",
                        color="gray"
                    ),
                    rx.button(
                        "Dosya Seç",
                        color_scheme="blue",
                        variant="soft",
                        margin_top="1em"
                    ),
                    align="center",
                ),
                id="file_upload",
                border="2px dashed var(--gray-6)",
                border_radius="md",
                padding="3em",
                multiple=True,
                accept={
                    "audio/*": [".wav", ".mp3", ".ogg"]
                },
                max_files=5,
                disabled=False,
                on_drop=State.handle_upload(rx.upload_files(upload_id="file_upload")),
                width="100%",
                _hover={"bg": "var(--gray-3)", "cursor": "pointer"},
            ),
            rx.cond(
                State.is_uploaded,
                rx.vstack(
                    rx.heading("Yüklenen Dosyalar", size="5", margin_top="1em"),
                    rx.foreach(
                        State.uploaded_files,
                        lambda filename: rx.card(
                            rx.vstack(
                                rx.text(filename, font_weight="bold", size="2"),
                                rx.audio(src=rx.get_upload_url(filename)),
                            ),
                            width="100%"
                        ),
                    ),
                    rx.button(
                        rx.cond(
                            State.is_inferring,
                            "İşleniyor...",
                            "Sınıflandır"
                        ),
                        on_click=State.run_inference,
                        disabled=~State.is_uploaded | State.is_inferring,
                        size="3",
                        width="100%",
                        color_scheme="green",
                        margin_top="1em",
                    ),
                    width="100%",
                    spacing="3",
                ),
                rx.text("Henüz dosya yüklenmedi.", color="gray", margin_top="1em")
            ),
            rx.cond(
                State.results,
                rx.vstack(
                    rx.heading("Sonuçlar", size="5", margin_top="1em"),
                    rx.foreach(
                        State.results,
                        lambda result: rx.badge(
                            result,
                            size="3",
                            color_scheme="blue",
                            radius="full",
                            padding="0.5em 1em"
                        ),
                    ),
                    width="100%",
                    spacing="3",
                    align="center",
                ),
            ),
            spacing="5",
            align="center",
            width="100%",
            max_width="600px",
            margin_x="auto",
            padding_y="4em",
        ),
    )


app = rx.App()
app.add_page(index)
