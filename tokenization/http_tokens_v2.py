import glob
import re
from typing import Counter
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

        if isinstance(value, str) and value.startswith("{") and "'" in value and '"' not in value:
            value = re.sub(r"'", '"', value)

        return json.loads(value)

    except (json.JSONDecodeError, TypeError):
        return []
    
def split_value(value):

    if value is None:
        return []

    if not isinstance(value, str):
        value = str(value)

    parts = re.split(r"[,\;\|\s\-\_]+", value)

    return [p.strip() for p in parts if p.strip()]

    
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
    
## TOKENIZAÇÃO ##
def tokenize_headers(headers_dict, KEYS_TO_IGNORE, KEYS_TO_GENERALIZE):
    tokens = []

    if not headers_dict:
        return ["H_is_absent"]

    for key, value in headers_dict.items():

        if key in KEYS_TO_IGNORE:
            continue

        if key in KEYS_TO_GENERALIZE:
            gen = KEYS_TO_GENERALIZE[key]
            tokens.append(f"H_{key}={gen}")
            continue

        if value is None or value == "":
            tokens.append(f"H_{key}=empty")
            continue

        parts = split_value(str(value))

        if len(parts) == 1:
            tokens.append(f"H_{key}={parts[0]}")
        else:
            for i, p in enumerate(parts, start=1):
                tokens.append(f"H_{key}_{i}={p}")

    return tokens

def tokenize_params(params_dict,  KEYS_TO_IGNORE, KEYS_TO_GENERALIZE):
    tokens = []

    if not params_dict:
        return ["P_is_empty"]

    for key, value in params_dict.items():

        if key in KEYS_TO_IGNORE:
            continue

        if key in KEYS_TO_GENERALIZE:
            tokens.append(f"P_{key}={KEYS_TO_GENERALIZE[key]}")
            tokens.append(f"P_{key}_length={len(KEYS_TO_GENERALIZE[key])}")
            continue

        if value is None or value == "":
            tokens.append(f"P_{key}=empty")
            tokens.append("P_{key}_length=0")
            continue

        value_str = str(value)
        parts = split_value(value_str)

        tokens.append(f"P_{key}_length={len(value_str)}")

        if len(parts) == 1:
            tokens.append(f"P_{key}={parts[0]}")
        else:
            for i, p in enumerate(parts, start=1):
                tokens.append(f"P_{key}_{i}={p}")

    return tokens

def create_vocabulary(df_requests):
    tokenized_sentences = []

    for _, row in df_requests.iterrows():
        request_sentence = []

        # ===== HEADERS =====
        headers_dict = safe_json_load(row.get("headers", {}))
        header_tokens = tokenize_headers(
            headers_dict=headers_dict,
            KEYS_TO_GENERALIZE=KEYS_TO_GENERALIZE,
            KEYS_TO_IGNORE=KEYS_TO_IGNORE
        )
        request_sentence.extend(header_tokens)

        # ===== PARAMS =====
        params_dict = safe_json_load(row.get("params", {}))
        param_tokens = tokenize_params(
            params_dict=params_dict,
            KEYS_TO_GENERALIZE=KEYS_TO_GENERALIZE,
            KEYS_TO_IGNORE=KEYS_TO_IGNORE
        )
        request_sentence.extend(param_tokens)

        # ===== IGNORA REQUESTS VAZIOS =====
        if request_sentence:
            tokenized_sentences.append(request_sentence)

    return tokenized_sentences