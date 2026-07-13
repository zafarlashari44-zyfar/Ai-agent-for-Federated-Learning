from datetime import datetime


class LoggerAgent:

    def log(self, message):

        now = datetime.now().strftime("%H:%M:%S")

        print(f"[{now}] {message}")