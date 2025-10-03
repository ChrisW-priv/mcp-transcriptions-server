from gtts import gTTS

tts = gTTS("hello world", lang="en")
tts.save("../tests/dummy_audio.mp3")
