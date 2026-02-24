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
from tqdm import tqdm

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
       'Cf-Region-Code', 'Cf-Timezone', 'X-Verified-Bot-Category',
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
    'oracle', 'alibaba', "tencent", "limstone", "cloudflare"
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

    if "Ip-Api-Duration" in ipapi_fields:
        try:
            duration = float(ipapi_fields["Ip-Api-Duration"])
            tokens.append(f"ipapi_duration_bucket={int(duration // 50)}") 
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
            if device == "Generic Smartphone":
                print(ua_string)
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
def define_params(params_dict, traffic_source):
    if traffic_source.lower() == "facebook": 
        PARAMS = FB_PARAMS
    elif traffic_source.lower() == "tiktok":
        PARAMS = TT_PARAMS
    elif traffic_source.lower() == "google":
        PARAMS = GG_PARAMS
    tokens = []

    if not params_dict:
        return ['p_is_empty']

    try:
        tokens.append(f'p_count={len(params_dict)}')

        for k, v in params_dict.items():
            # Detevta valores padrão incorretos
            if v == PARAMS.get(k):
                tokens.append(f"P_{k}={v}")
            elif not PARAMS.get(k):
                continue
            else:
                if k in KEYS_TO_GENERALIZE:
                    value = KEYS_TO_GENERALIZE.get(k)
                    tokens.append(f'P_{k}={value}')

        return tokens

    except Exception as e:
        return ['p_parser_error']


# def define_params(params_dict):
#     tokens = []

#     if not params_dict:
#         return ['PARAMS_is_empty']
    
#     try:
#         tokens.append(f'P_count={len(params_dict)}')

#         # Verifica se há parâmetros conhecidos (baseados em TT_PARAMS)
#         for key in TT_PARAMS.keys():
#             tokens.append(f"P_has_{key}={int(key in params_dict)}")

#         misconfig_params = []

#         # --- Loop principal: análise por parâmetro individual ---
#         for k, v in params_dict.items():
#             # Verifica se o parâmetro está com valor padrão
#             if v == TT_PARAMS.get(k):
#                 tokens.append(f"P_{k}_is_default=True")
#                 misconfig_params.append(k)
#             elif isinstance(v, str) and "__" in v:
#                 tokens.append(f"P_{k}_looks_placeholder=True")
#             else:
#                 tokens.append(f'P_{k}_is_default=False')

#         # --- Flags agregadas (uma única vez) ---
#         if misconfig_params:
#             tokens.append("P_has_bad_config=True")
#             tokens.append(f"P_bad_keys={len(misconfig_params)}")
#         else:
#             tokens.append("P_has_bad_config=False")

#         # --- Indicadores de tracking / marketing (uma vez) ---
#         if 'utm_source' in params_dict:
#             tokens.append(f'P_source={params_dict["utm_source"].lower()}')

#         tokens.append(f'P_has_utm_medium={int("utm_medium" in params_dict)}')

#         if any("ttclid" in k.lower() for k in params_dict.keys()):
#             tokens.append('P_has_ttclid=True')
#         else:
#             tokens.append('P_has_ttclid=False')

#         return tokens
    
#     except Exception as e:
#         return ['P_parser_error']

        
def create_vocabulary(data, traffic_source=None):
    corpus = []

    for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Criando Vocabulário"):

    # for index, row in data.iterrows():
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
                    token = f"H_{key}={KEYS_TO_GENERALIZE[key]}"
                else:
                    token = f"H_{key}={str(value)}" 
                
                request_sentence.append(token)
                
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
                
                if traffic_source:
                    params_token = define_params(params_dict, traffic_source)
                    request_sentence.extend(params_token)
                    
            if request_sentence: 
                corpus.append(request_sentence)
                
        except json.JSONDecodeError:
            print(f"Erro ao processar {index}")
            continue
    return corpus


def create_request_vector(tokens, model):
    vectors = []

    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])

    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)

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