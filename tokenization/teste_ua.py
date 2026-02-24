import re

def parse_user_agent_raw(ua_string):

    if not ua_string:
        return ['ua_is_empty']

    # quebra tudo em palavras alfanuméricas + hífens e underscores
    raw_tokens = re.findall(r"[A-Za-z0-9\-\_\.]+", ua_string)

    # limpa e normaliza
    tokens = []
    for t in raw_tokens:
        t = t.strip(" .,_;:()[]{}")
        if not t:
            continue
        # evita duplicatas simples e palavras curtas demais
        if len(t) > 1 and t.lower() not in ("mozilla", "compatible"):
            tokens.append(f"ua_{t}")
    return tokens

if __name__ == "__main__":
    from user_agents import parse

    ua_string = "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
    ua = parse(ua_string)

    print("Device:", ua.device.family)
