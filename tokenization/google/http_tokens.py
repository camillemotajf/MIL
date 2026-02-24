import re
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import json
from sklearn.discriminant_analysis import StandardScaler
from user_agents import parse
from urllib.parse import urlparse
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

KEYS_TO_IGNORE = ['User-Agent', 'Accept', 'X-Forwarded-For', 'X-Original-Forwarded-For',
       'X-Forwarded-Port', 'Forwarded', 'X-Scheme', 'X-Forwarded-Proto',
       'X-Forwarded-Scheme', 'Cdn-Loop', 'Cf-Ray', 'Cf-Connecting-O2o',
       'Host', 'Ip-Api-Duration', 'Ip-Api-City', 'Ip-Api-Zip',
       'Ip-Api-Asname', 'Ip-Api-As', 'Ip-Api-Org', 'Ip-Api-Isp',
       'Ip-Api-Region-Name', 'Ip-Api-Hosting', 'Ip-Api-Proxy',
       'Cf-Visitor', 'Cf-Connecting-Ip', 'Cf-Ipcountry',
       'X-Forwarded-Host', 'Referer', 'X-Request-Id', 'Accept-Encoding',
       'Sec-Ch-Ua', 'Sec-Fetch-Mode', 'Sec-Fetch-Size',
       'X-Browser-Copyright', 'X-Bluecoat-Via', 'X-Browser-Year',
       'X-Proxyuser-Ip', 'Cf-Ew-Via', 'Via', 'X-Imforwards',
       'Sec-Fetch-User', 'Upgrade-Insecure-Requests', 'Sec-Gpc',
       'If-Modified-Since', 'X-Browser-Validation', 'Prefer',
       'X-Browser-Channel', 'If-None-Match', 'X-Ucbrowser-Ua',
       'X-Referer', 'Dnt', 'Sec-Purpose', 'X-Asn', 'X-Enrichmentstatus',
       'Cf-Region-Code', 'Cf-Timezone',
       'Cookie', 'Cf-Ipcity', 'Cf-Region', 'Origin']

KEYS_TO_GENERALIZE = {
    'X-Request-Id': '_ID_',
    'X-Real-Ip': '_IP_',
    'Cf-Ray': '_ID_',
    'Cf-Connecting-Ip': '_IP_',
    "fbclid" : "CLID",
    "utm_id": "ID",
    "click_id": "ID",
    "src":"SRC",
    "utm_medium":"ID",
    "utm_content":"ID",
    "xid":"ID",
    "ttclid":"ID",
    "utm_campaign": "name",
    "cname": "name",
    "cwr": "id"
}

ASNAME_CLOUD_KEYWORDS = [
    'google', 'amazon', 'aws', 'microsoft', 'azure', 
    'digitalocean', 'linode', 'ovh', 'hetzner', 'scaleway', 
    'oracle', 'alibaba', "tencent", "limstone", "cloudflare", "akamai"
]

ASNAME_PROXY_KEYWORDS = [
    'proxy', 'vpn', 'fwdproxy', 'cache', 'tor'
]

## Pamaetros de url
TT_PARAMS = {
    "src":"__CSITE__",
    "utm_medium":"__AID_NAME__",
    'utm_content':"__CID_NAME__",
    "sku":"__PLACEMENT__",
    "click_id":"__CALLBACK_PARAM__",
    "xid":"HASH"
}

GG_PARAMS = {
    "cr": "{creative}",
    "plc": "{placement}",
    "mtx": "{matchtype}",
    "rdn": "{random}",
    "kw": "{keyword}",
    "gclid": "{gclid}",
    "wbraid": "{wbraid}",
    "gbraid": "{gbraid}",
    "xid": "99cd51yl"
}

FB_PARAMS = {
    "cwr":"{{campaign.id}}",
    "cname":"{{campaign.name}}",
    "domain":"{{domain}}",
    "placement":"{{placement}}",
    "adset":"{{adset.name}}",
    "adname":"{{ad.name}}",
    "site":"{{site_source_name}}",
}


## browser suspeitos
SUSPICIOUS_browserS = [
    "bot", "crawler", "spider", "google", "bing", "tiktok", "twitter", "facebook", "yahoo"
]

def get_browser_sus_tokens(browser_string):
    tokens = []

    if not browser_string:
        return ["browser_is_empty"]
    
    browser_lower = browser_string.lower().replace(" ", "")

    for key in SUSPICIOUS_browserS:
        if key in browser_lower:
            tokens.append(f"browser_sus={key}")

    if not tokens:
        tokens.append("browser_normal")
    
    return tokens


def pca_redution(X, n=10, data=None):

    pca_scaler = StandardScaler(with_mean=False)
    X_scaled = pca_scaler.fit_transform(X)

    pca = PCA(n_components=n, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)

    explained_variance = pca.explained_variance_ratio_
    cum_var = np.cumsum(explained_variance)

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(explained_variance)+1), cum_var, marker='o')
    plt.title('Variância Explicada Acumulada pelo PCA')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Variância Explicada Acumulada')
    plt.grid()
    plt.show()

    if data:
        X_pca = X_reduced  
        labels = data['decision']

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')

        for classe in np.unique(labels):
            idx = labels == classe
            ax.scatter(X_pca[idx, 0], X_pca[idx, 1], X_pca[idx, 2], label=classe, s=10)

        ax.set_title("Dados em 3 Componentes Principais (PCA)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.legend()
        plt.ion()

        plt.show()

    return X_reduced

def safe_json_load(value):
    if value in [None, "", []]:
        return []

    try:
        # Já é dict ou list
        if isinstance(value, (dict, list)):
            return value

        # Corrige aspas simples → duplas
        if isinstance(value, str) and value.startswith("{") and "'" in value and '"' not in value:
            value = re.sub(r"'", '"', value)

        return json.loads(value)

    except (json.JSONDecodeError, TypeError):
        return []
    
def get_asname_tokens(asname_string):
    """
    Gera tokens a partir do campo 'Ip-Api-Asname':
    - Classifica tipo (cloud_hosting, proxy_vpn, isp_residential)
    - Cria tokens de palavras individuais (ex: asname_word=amazon)
    """
    if not asname_string:
        return ['asname_is_empty']
    
    tokens = []
    asname_lower = asname_string.lower().replace('-', ' ').replace('_', ' ')
    as_words = [w.strip() for w in asname_lower.split() if w.strip()]
    
    # Tokens individuais de palavras
    for word in as_words:
        tokens.append(f"asname_word={word}")
    
    # Classificação por palavras-chave
    found_category = False
    
    for kw in ASNAME_CLOUD_KEYWORDS:
        if kw in asname_lower:
            tokens.append('asname_type=cloud_hosting')
            tokens.append(f'asname_provider={kw}')
            found_category = True
            break
    
    if not found_category:
        for kw in ASNAME_PROXY_KEYWORDS:
            if kw in asname_lower:
                tokens.append('asname_type=proxy_vpn')
                tokens.append(f'asname_provider={kw}')
                found_category = True
                break
    
    if not found_category:
        tokens.append('asname_type=isp_residential')
    
    return tokens

def define_ipapi_tokens(headers_dict):
    """
    Extrai e tokeniza informações do Ip-Api-* para criação de features semânticas.
    """
    tokens = []

    if not headers_dict:
        return ["ipapi_is_empty"]

    # Mapeia todos os campos Ip-Api-* disponíveis
    ipapi_fields = {k: v for k, v in headers_dict.items() if k.lower().startswith("ip-api-")}
    if not ipapi_fields:
        return ["ipapi_not_present"]

    # --------- IP-Api-Asname ----------
    asname = ipapi_fields.get("Ip-Api-Asname")
    if asname:
        tokens.append(f"ipapi_asname_raw={asname.lower().replace(' ', '_')}")
        tokens.extend(get_asname_tokens(asname))

        # Quebra o nome em palavras individuais (ex: "tencent-net-ap-cn")
        parts = re.split(r"[\s\-_]+", asname.lower())
        tokens.extend([f"asname_word={p}" for p in parts if len(p) > 1])

    # --------- IP-Api-As ----------
    as_value = ipapi_fields.get("Ip-Api-As")
    if as_value:
        # Ex: "as132203 tencent building, kejizhongyi avenue"
        parts = re.split(r"[\s,\-_]+", as_value.lower())
        tokens.append(f"ipapi_as_id={parts[0]}" if parts else "ipapi_as_id=unknown")
        tokens.extend([f"as_word={p}" for p in parts if len(p) > 1 and not p.startswith("as")])

        # Detecta se há palavras-chave conhecidas (cloud, vpn, etc.)
        for kw in ASNAME_CLOUD_KEYWORDS:
            if kw in as_value.lower():
                tokens.append("as_type=cloud_hosting")
                tokens.append(f"as_provider={kw}")
                break
        else:
            tokens.append("as_type=isp_residential")

    # --------- ORG / ISP ----------
    org = ipapi_fields.get("Ip-Api-Org")
    isp = ipapi_fields.get("Ip-Api-Isp")

    for field, label in [(org, "org"), (isp, "isp")]:
        if field:
            parts = re.split(r"[\s,\-_]+", field.lower())
            tokens.extend([f"{label}_word={p}" for p in parts if len(p) > 1])
            for kw in ASNAME_CLOUD_KEYWORDS:
                if kw in field.lower():
                    tokens.append(f"{label}_type=cloud_hosting")
                    tokens.append(f"{label}_provider={kw}")
                    break

    # --------- PROXY / HOSTING / LOCATION ----------
    if "Ip-Api-Proxy" in ipapi_fields:
        tokens.append(f"ipapi_proxy={ipapi_fields['Ip-Api-Proxy']}")
    if "Ip-Api-Hosting" in ipapi_fields:
        tokens.append(f"ipapi_hosting={ipapi_fields['Ip-Api-Hosting']}")
    if "Ip-Api-City" in ipapi_fields:
        tokens.append(f"ipapi_city={ipapi_fields['Ip-Api-City'].lower().replace(' ', '_')}")
    if "Ip-Api-Region-Name" in ipapi_fields:
        tokens.append(f"ipapi_region={ipapi_fields['Ip-Api-Region-Name'].lower().replace(' ', '_')}")
    if "Cf-Ipcountry" in headers_dict:
        tokens.append(f"ipapi_country={headers_dict['Cf-Ipcountry'].lower()}")

    # --------- Duração (opcional, se quiser tratar como numérica) ----------
    if "Ip-Api-Duration" in ipapi_fields:
        try:
            duration = float(ipapi_fields["Ip-Api-Duration"])
            tokens.append(f"ipapi_duration_bucket={int(duration // 50)}")  # bucket por intervalos de 50ms
        except Exception:
            tokens.append("ipapi_duration_bucket=unknown")

    return tokens


## Processamento do User-Agent
def get_ua_tokens(ua_string):

    if not ua_string:
        return ['ua_is_empty']

    try:
        ua = parse(ua_string)
        tokens = []

        if ua.os.family:
            os = ua.os.family.replace(' ', '')
            tokens.append(f'ua_os={os}')
        if ua.os.version_string:
            os_family = ua.os.version_string
            tokens.append(f"ua_os_version={os_family}")
            
        if ua.device.family:
            device = ua.device.family.replace(' ', '')
            tokens.append(f'ua_device={device}')

        if ua.is_bot:
            tokens.append('ua_is_bot=True')
            if ua.browser.family:
                browser = ua.browser.family
                tokens.append(f'ua_bot_family={ua.browser.family.replace(" ", "")}')

                sus_tokens = get_browser_sus_tokens(browser)
                sus_score = sum("ua_sus" in t for t in sus_tokens)

                if sus_score > 0:
                    tokens.append(f"ua_sus_score={sus_score*5}")    

                tokens.extend(sus_tokens)
        else:
            tokens.append('ua_is_bot=False')

        if ua.is_mobile:
            tokens.append('ua_device_type=mobile')
        elif ua.is_tablet:
            tokens.append('ua_device_type=tablet')
        elif ua.is_pc:
            tokens.append('ua_device_type=pc')
        else:
            tokens.append('ua_device_type=unknown')

        if ua.browser.family:
            # print(type(ua.browser.family))
            browser = ua.browser.family.replace(' ', '')

            browser_family = ua.browser.version_string

            tokens.append(f'ua_browser={browser}')

            if browser_family:
                tokens.append(f"ua_browser_family={browser_family}")

            sus_tokens = get_browser_sus_tokens(browser)
            sus_score = sum("ua_sus" in t for t in sus_tokens)

            if sus_score > 0:
                tokens.append(f"ua_sus_score={sus_score*5}")    

            tokens.extend(sus_tokens)

        return tokens
    
    except Exception as e:
        return ['ua_parse_error']

## teste de utilização de ua
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
    

## Processamento do Accept Language

def get_lang_tokens(lang_string):

    if not lang_string:
        return ['lang_is_empty']
    
    tokens = []

    try:
        parts = [part.strip() for part in lang_string.split(',')]
        tokens.append(f"lang_count={len(parts)}")

        if any(';q=' in part for part in parts):
            tokens.append('lang_has_q=True')
        else:
            tokens.append('lang_has_q=False')
        
        primary_part = parts[0]
        primary_lang_code = primary_part.split(';')[0]
        tokens.append(f'lang_primary={primary_lang_code}')

        return tokens

    except Exception as e:
        return ['lang_parse_error']
    

## Processamento do accept:
def get_accept_tokens(accept_string):
    if not accept_string:
        return ['accept_is_empty']
    
    tokens = []
    lower_string = accept_string.lower()

    try:
        parts = [part.strip() for part in lower_string.split(",")]
        tokens.append(f'accept_count={len(parts)}')

        if any(';q=' in part for part in parts):
            tokens.append('accept_has_q=True')
        else:
            tokens.append('accept_has_q=False')

        if '*/*' in lower_string:
            tokens.append('accept_has_wildcard=True')
            
        if 'image/avif' in lower_string or 'image/webp' in lower_string:
            tokens.append('accept_supports_modern_img=True')
        
        if 'text/html' in lower_string:
            tokens.append('accept_supports_html=True')
        
        if accept_string == '*/*':
            tokens.append('accept_is_only_wildcard')

        return tokens
    except Exception as e:
        return ["accept_parse_error"]


## processando accept encoding
def get_encoding_tokens(encoding_string):

    if not encoding_string:
        return ['encoding_is_empty'] 
        
    tokens = []
    
    try:
        normalized_string = encoding_string.lower().replace(' ', '')
        parts = [part.split(';')[0] for part in normalized_string.split(',')]
        tokens.append(f'encoding_count={len(parts)}')
        
        if 'gzip' in parts:
            tokens.append('encoding_has_gzip=True')
        
        if 'br' in parts:
            tokens.append('encoding_has_brotli=True')
        
        if 'deflate' in parts:
            tokens.append('encoding_has_deflate=True')
            
        if '*' in parts:
            tokens.append('encoding_has_wildcard=True')
        
        if normalized_string == '*':
            tokens.append('encoding_is_only_wildcard')

        return tokens
        
    except Exception as e:
        return ['encoding_parse_error']
    

## Referer
def get_referer_tokens(referer_string, host_string):

    if not referer_string:
        return ["referer_is_empty"]
    
    tokens = []

    try:
        parsed_uri = urlparse(referer_string)
        referer_domain = parsed_uri.hostname

        if host_string in referer_string:
            tokens.append("referer_has_host=True")
        else:
            tokens.append("referer_has_host=False")

        if not referer_domain:
            return ["referer_domain_is_empty"]
        
        return tokens
        
    except Exception as e:
        return ["referer_parse_error"]

# parametros de url   
def define_params(params_dict, traffic_source=None):
    if not traffic_source: 
        PARAMS = {**FB_PARAMS, **TT_PARAMS, **GG_PARAMS}
    elif traffic_source.lower() == "tiktok":
        PARAMS = TT_PARAMS
    elif traffic_source.lower() == "google":
        PARAMS = GG_PARAMS
    elif traffic_source.lower() == "facebook":
        PARAMS = FB_PARAMS
    
    tokens = []

    if not params_dict:
        return ['p_is_empty']

    try:
        tokens.append(f'p_count={len(params_dict)}')

        # Loop apenas para tokens específicos de cada parâmetro
        for k, v in params_dict.items():
            if k in PARAMS:
                # Detevta valores padrão incorretos
                if v == PARAMS.get(k):
                    tokens.append(f"p_{k}={v}")
                elif not PARAMS.get(k):
                    continue
                else:
                    if k in KEYS_TO_GENERALIZE:
                        value = KEYS_TO_GENERALIZE.get(k)
                        tokens.append(f'p_{k}={value}')

        return tokens

    except Exception as e:
        return ['p_parser_error']


def smart_tokenize_header(key, value):

    if value is None or value == "":
        return [f"H_{key}=empty"]

    s = str(value)

    separators = r"[\-_\./\|\:\?\&\=]"
    parts = re.split(separators, s)
    parts_clean = [p for p in parts if len(p) > 1]

    too_many_parts = len(parts_clean) >= 3
    has_separators = bool(re.search(separators, s))
    is_long = len(s) > 25
    is_hashlike = bool(re.search(r"[a-zA-Z]+[0-9]+", s)) or bool(re.search(r"[0-9]+[a-zA-Z]+", s))

    if not (too_many_parts or has_separators or is_long or is_hashlike):
        return [f"H_{key}={s}"]

    tokens = [f"H_{key}_tok={p.lower()}" for p in parts_clean[:5]]
    tokens.append(f"H_{key}_count={len(parts_clean)}")

    return tokens


def tokenize_sec_header(key, value):

    tokens = []
    base = f"{key.lower()}"

    if value is None or value == "" or str(value).strip() == "":
        tokens.append(f"{base}=empty")
        tokens.append(f"{base}_class=suspicious")
        return tokens

    v = str(value).lower().strip()
    tokens.append(f"{base}={v}")

    # Sec-Fetch-Dest
    if key.lower() == "sec-fetch-dest":
        if v == "document":
            tokens.append(f"{base}_type=document")
        elif v == "image":
            tokens.append(f"{base}_type=image")
        elif v == "empty":
            tokens.append(f"{base}_type=empty")
            # bots usam muito "empty"
            tokens.append(f"{base}_class=botlike")
        else:
            tokens.append(f"{base}_type=other")

    # Sec-Fetch-Site
    elif key.lower() == "sec-fetch-site":
        if v == "same-origin":
            tokens.append(f"{base}_site=same_origin")
        elif v == "same-site":
            tokens.append(f"{base}_site=same_site")
        elif v == "none":
            tokens.append(f"{base}_site=none")
        elif v == "cross-site":
            tokens.append(f"{base}_site=cross_site")
            tokens.append(f"{base}_class=suspicious")  # bots usam muito
        else:
            tokens.append(f"{base}_site=other")

    # Sec-Fetch-Mode
    elif key.lower() == "sec-fetch-mode":
        tokens.append(f"{base}_mode={v}")

    # Sec-Fetch-User
    elif key.lower() == "sec-fetch-user":
        if v == "?1":
            tokens.append(f"{base}_bool=true")
        elif v == "?0":
            tokens.append(f"{base}_bool=false")
        else:
            tokens.append(f"{base}_bool=unknown")

    # Sec-CH-UA (client hints)
    elif key.lower() == "sec-ch-ua":
        # quebra o valor por vírgulas, sem destruir user-agent
        parts = re.findall(r'"([^"]+)"', v)
        for p in parts[:5]:
            tokens.append(f"{base}_brand={p.lower()}")

    # Sec-CH-Mobile
    elif key.lower() == "sec-ch-mobile":
        if v == "?1":
            tokens.append(f"{base}_mobile=true")
        elif v == "?0":
            tokens.append(f"{base}_mobile=false")
        else:
            tokens.append(f"{base}_mobile=unknown")

    # Sec-CH-UA-Platform
    elif key.lower() == "sec-ch-ua-platform":
        tokens.append(f"{base}_platform={v.strip('\"')}")

    if key.lower().startswith("sec-"):

        if len(v) > 40:
            tokens.append(f"{base}_class=suspicious_long")

        if re.match(r"^[a-z0-9]{20,}$", v):
            tokens.append(f"{base}_class=hashlike")

        parts = re.split(r"[ \-_/;:]+", v)
        parts = [p for p in parts if len(p) > 1]

        for p in parts[:5]:
            tokens.append(f"{base}_tok={p}")

        tokens.append(f"{base}_tokcount={len(parts)}")

    return tokens

        
def create_vocabulary(data, traffic_source=None):
    corpus = []
    for index, row in data.iterrows():
        request_sentence = []
        
        try: 
            headers_dict = safe_json_load(row['headers'])

            if not isinstance(headers_dict, dict):
                print("erro")

            ua_string = headers_dict.get('User-Agent')
            ua_tokens = get_ua_tokens(ua_string)      
            request_sentence.extend(ua_tokens)

            # ua_tokens_raw = parse_user_agent_raw(ua_string)
            # request_sentence.extend(ua_tokens_raw)
            
            # ua_tokens = parse_user_agent_raw(ua_string)      
            # request_sentence.extend(ua_tokens)

            ## Accept Language
            lang_string = headers_dict.get('Accept-Language') 
            lang_tokens = get_lang_tokens(lang_string)     
            request_sentence.extend(lang_tokens)

            ## Accept
            accept_string = headers_dict.get('Accept') 
            accept_tokens = get_accept_tokens(accept_string)     
            request_sentence.extend(accept_tokens)

            ## Accept Encoding
            encoding_string = headers_dict.get('Accept-Encoding') 
            encoding_tokens = get_encoding_tokens(encoding_string) 
            request_sentence.extend(encoding_tokens)

            ## Sec headers
            for key, value in headers_dict.items():
                if key.lower().startswith("sec-"):
                    print(tokenize_sec_header(key, value))
                    request_sentence.extend(tokenize_sec_header(key, value))
                    print(request_sentence)
                    exit()
                    continue


            # asname_string = headers_dict.get('Ip-Api-Asname')
            # asname_tokens = get_asname_tokens(asname_string)
            # request_sentence.extend(asname_tokens)

            ipapi_tokens = define_ipapi_tokens(headers_dict)
            request_sentence.extend(ipapi_tokens)

            ## Referer
            referer = headers_dict.get('Referer')
            if referer == None:
                referer_tokens = ["referer_is_absent"]
            else:
                referer_string = headers_dict.get('Referer') 
                host_string = headers_dict.get("Host")
                referer_tokens = get_referer_tokens(referer_string, host_string) 

            request_sentence.extend(referer_tokens)


            for key, value in headers_dict.items():
                if key in KEYS_TO_IGNORE:
                    continue

                if key in KEYS_TO_GENERALIZE:
                    request_sentence.append(f"H_{key}={KEYS_TO_GENERALIZE[key]}")
                    continue

                tokens = smart_tokenize_header(key, value)
                request_sentence.extend(tokens)
                
            params_dict = safe_json_load(row['params'])

            if isinstance(params_dict, list):
                request_sentence.append(f'p_is_empty')
            elif isinstance(params_dict, dict):
                for k, v in params_dict.items():
                    if k in KEYS_TO_IGNORE:
                        continue
                
                    if k in KEYS_TO_GENERALIZE:
                        params_token = f"p_{k}={KEYS_TO_GENERALIZE[k]}"
                    else:
                        # params_token = define_params(params_dict)
                        params_token = f"p_{k}=key"
                    
                    request_sentence.append(params_token)
                
                params_token = define_params(params_dict, traffic_source)
                request_sentence.extend(params_token)
                    
            if request_sentence: 
                corpus.append(request_sentence)
                
        except json.JSONDecodeError:
            print(f"Erro ao processar {index}")
            continue
    return corpus


def create_request_vector(tokens, model):

    vector_size = model.vector_size
    
    request_vec = np.zeros(vector_size).astype('float32')
    
    num_words_in_vocab = 0
    
    for token in tokens:
        if token in model.wv:
            request_vec = np.add(request_vec, model.wv[token])
            num_words_in_vocab += 1
            
    if num_words_in_vocab > 0:
        request_vec = np.divide(request_vec, num_words_in_vocab)
        
    return request_vec

def create_tfidf_vector(tokens, model, tfidf_vectorizer):
    vector_size = model.vector_size
    request_vec = np.zeros(vector_size).astype('float32')

    tfidf_vocab = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))

    weights = []
    vectors = []

    for token in tokens:
        if token in model.wv:
            weight = tfidf_vocab.get(token, 1.0)
            vectors.append(model.wv[token] * weight)
            weights.append(weight)

    if weights:
        request_vec = np.sum(vectors, axis=0) / np.sum(weights)
    
    return request_vec


def has_bot(header_str):
    try:
        if not header_str or header_str == "None":
            return False

        # se já for dict
        if isinstance(header_str, dict):
            header = header_str
        # se for string JSON
        elif isinstance(header_str, str):
            try:
                header = json.loads(header_str)
            except json.JSONDecodeError:
                return False
        else:
            return False

        ua = header.get("User-Agent", "")
        if not isinstance(ua, str):
            return False

        ua_clean = ua.lower().replace(" ", "")
        # print(ua_clean)

        suspicious_words = [
            "facebookbot"
        ]

        if any(word in ua_clean for word in suspicious_words):
            print(ua_clean)

        return any(word in ua_clean for word in suspicious_words)

    except Exception:
        return False