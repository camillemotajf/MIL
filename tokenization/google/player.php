<?php
header("Access-Control-Allow-Origin: *");
header("Content-Type: text/html; charset=UTF-8");
header("Cache-Control: no-cache, must-revalidate"); // HTTP 1.1.
header("Pragma: no-cache"); // HTTP 1.0.
header("Expires: Sat, 26 Jul 1997 05:00:00 GMT"); // Data no passado

/* START OF SCRIPT SETTINGS */
require "config.php";

$UNSAFE_SITE_REDIRECT_URL = null;

// If pointing to a local file, will print the contents of that file instead of
// redirecting. If null, will redirect to $UNSAFE_SITE_REDIRECT_URL
$UNSAFE_FILE = UNSAFE_FILE; // ex: null, "./unsafe.html"

// If set to null or "", the cloaker will return the html content from this
// file. Else it will redirect to the specified url with a 302 http redirect.
$SAFE_SITE_REDIRECT_URL = null; // ex: null, "", "http://www.google.com"

// If pointing to a local file, will print the contents of that file instead of
// redirecting. If null, will redirect to $SAFE_SITE_REDIRECT_URL
$SAFE_FILE = "./safe.php"; // ex: null, "./safe.html"

// An id to help identify each site on the database
$site_identification = "###SITE_IDENTIFICATION###";

// Some parameters can be configured with variables on the traffic source. For
// example: utm_source={source}. These variables are filled by the traffic
// source. However, when the traffic source accesses the url for checking the
// page content, it might leave these variables unfilled. So, a possible way
// of detecting a bot can be just checking if some url parameters contain
// unfilled variables as values.

// Set to true to send requests whose url parameters contain token chars to
// the safe page.
$filter_by_token = true; // true or false
// Url parameters that should be checked for token characters.
$token_parameters = "cr"; // ex: "utm_campaign,utm_source"
// Characters that identify if a value is a token.
$token_chars = "{}"; // ex: "[]{}"

// Allows filtering specific parameters by value.
// If whitelist, the list of parameters provided in $filter_parameters should
// contain values in whitelist. If blacklist, the list of parameters provided in
// $filter_parameters should contain values in blacklist. If false, this filter
// is ignored.
// You can also set $parameter_safe_if_missing to send to SAFE if any parameter
// in $filter_parameters is missing, and $parameter_safe_if_empty to send to
// SAFE if any parameter in $filter_parameters is empty.
$filter_by_parameter = "blacklist"; // "whitelist", "blacklist" or false
$filter_parameters = "cr";
$parameter_whitelist = "unsafe,facebook,adwords,gemini"; // ex: "cod123,xyz123"
$parameter_blacklist = "0"; // ex: "BR,AR,US"
// if true, if the parameter is absent from the url redirects to SAFE.
$parameter_safe_if_absent = false;
// if true, if the parameter has empty value redirects to SAFE.
$parameter_safe_if_empty = false;

// https://dev.maxmind.com/geoip/legacy/codes/iso3166/
$filter_by_country = "whitelist"; // "whitelist", "blacklist" or false
$country_whitelist = COUNTRY_WHITELIST; // ex: "BR,AR,US"
$country_blacklist = "US,IN,IE"; // ex: "BR,AR,US"

$filter_by_device = "whitelist";  // "whitelist", "blacklist" or false
$device_whitelist = DEVICE_WHITELIST; // ex: "mobile,desktop,unknown_device"
$device_blacklist = "unknown_device"; // ex: "desktop"

$filter_by_src = false; // "whitelist", "blacklist" or false
$src_whitelist = "unsafe,facebook,gemini,teste,adcash,adwords"; // ex: "cod123,xyz123"
$src_blacklist = ""; // ex: "BR,AR,US"


// You can filter also by ip address (both ipv4 and ipv6 addresses are
// supported). As usual, you can set a whitelist, a blacklist or you can disable
// ip filtering.
// Different than the other options, though, the whitelist and blacklist should
// be passed on files instead of direct strings.
// The files should be plain text files where each line will contain an ipv4 or
// ipv6 address. The ipv6 addresses should be EXPANDED (that is, abbreviations
// such as :1:123 or ::123 should be represented as :0001:0123 or
// :0000:0000:0123 for example). You can also replace sections of the ip by *
// in order to allow any value in that section (for example: 192.*.*.1). You can
// also specify only a portion of the ip address. In that case, that portion
// will be searched in any location of the ip address. For example: "face:b00c"
// would match with the ipv6 address "face:b00c::0001" and with the address
// "0001::face:b00c"
$filter_by_ip = "blacklist"; // "whitelist", "blacklist" or false
$ip_whitelist_file = "ip_whitelist.txt"; // ex: "ip_whitelist.txt"
$ip_blacklist_file = "ip_blacklist.txt"; // ex: "ip_blacklist.txt"


// fbclid is an identifier added by Facebook in any link that is shared there.
// The absence of fbclid in the query params may indicate a bot it trying
// to access the link. If set to true, the cloaker will change fbclid to 0
$filter_by_fbclid = false;
// A string for which to replace the user's fbclid. If set to null the fbclid
// will not be replaced.
$fbclid_replace = '0';  // ex: null, "", "RANDOM_STRING"

// The referer header might indicate a bot is trying to access the cloaker's
// destination. Here you can set a blacklist or whitelist. The terms in the
// lists will be searched using a partial search: if the referer contains the
// black/whitelisted term it will return safe/unsafe accordingly.
$filter_by_referer = false; // "whitelist", "blacklist" or false
$referer_whitelist = "youtube,youtube.com"; // ex:"facebook"
$referer_blacklist = "https://www.google.com/url?,http://www.ads.google.com/,https://tpc.googlesyndication.com/,embed,https://www.google.com,adclick,google.com,news.google.com";

// If true, redirects to SAFE if there isnt a referer on the header.
$referer_safe_if_absent = false;

// If true, redirects to SAFE if the Referer is the first field on the received
// headers. We think Facebook bots always come with Referer first.
$referer_safe_if_first = false;

$should_set_unsafe_cookie = false; // if true and if the user is allowed
    // to see unsafe content, a cookie will be set, so in future accesses we
    // can know that the user can be directly redirected to unsafe content
    // even if the unsafe conditions are not longer met.
$should_check_for_unsafe_cookie_presence = false; // if true, checks if the
    // unsafe cookie is set. Else, ignores the cookie.
$unsafe_cookie_name = 'sd'; // name of the cookie to check
$redirect_to_url_if_unsafe_cookie = null; // set to null if the presence of the
    // cookie should not redirect to a specific url. If null, the rules
    // defined on the index.php file will be respected. If not null, the
    // cloaker will perform a redirection to the specified URL overriding
    // any other behavior.
$return_pixel = false;

$use_ip_api = true; // if true, will perform a call to the ip-api
// (https://members.ip-api.com) and will make available the headers
// ip_api_proxy (1 or 0), ip_api_isp (a string) and ip_api_duration (duration
// in ms). These can be used in rules for filtering.
/* END OF SCRIPT SETTINGS */

// database connection settings
$save_to_database = true; // saves each user access to a local database.(Always the last that works)
$registered_page = "pixel";
$traffic_source = "google_shadow";
/* END OF SCRIPT SETTINGS */


$rules = [
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(User-Agent)/i', // rule for filtering User-Agent and user-agent
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*google.*)|(.*support.*)|(.*Trident.*)|(.*HeadlessChrome.*)|(.*googleweblight.*)|(.*instabridge.*)|(.*trailer.*)/i'
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(ip-api-isp|Ip-Api-Isp|Ip-Api-Org)/', // can be a string or a regular expression enclosed in / /
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*google.*)|(.*cache.*)|(.*proxy.*)|(.*youtube.*)|(.*datacenter.*)|(.*unknown.*)|(.*digitalocean.*)|(.*pubmatic.*)|(.*vultr.*)|(.*xlhdns.*)|(.*addr.*)|(.*amazon.*)|(.*akamai.*)|(.*boost mobile.*)|(.*packethub.*)|(.*clouvider.*)|(.*isomedia.*)|(.*hawaiian.*)|(.*level 3.*)|(.*university.*)/'
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => false,
        'key' => '/(ip-api-proxy|Ip-Api-Proxy)/', // can be a string or a regular expression enclosed in / /
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => 1
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => false,
        'key' => '/(ip-api-hosting|Ip-Api-Hosting)/', // can be a string or a regular expression enclosed in / /
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => 1
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(ip-api-reverse|Ip-Api-Reverse)/', // can be a string or a regular expression enclosed in / /
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*google.*)|(.*cache.*)|(.*proxy.*)|(.*youtube.*)|(.*datacenter.*)|(.*unknown.*)|(.*digitalocean.*)|(.*pubmatic.*)|(.*vultr.*)|(.*xlhdns.*)|(.*addr.*)|(.*amazon.*)|(.*akamai.*)/'
    ],
    [
        'type' => 'url_param', // 'header' or 'url_param'
        'enabled' => false,
        'key' => 'domain', // can be a string or a regular expression enclosed in / /
        'safe_if_absent' => false,
        'safe_if_empty' => true,
        'safe_if_present' => false,
        'filter_type' => 'whitelist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '/.*domain.*/',
        'blacklist' => ''
    ],
    [
        'type' => 'url_param', // 'header' or 'url_param'
        'enabled' => true,
        'key' => 'gclid', // can be a string or a regular expression enclosed in / /
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*gclid.*)/'
    ],
    [
        'type' => 'url_param', // 'header' or 'url_param'
        'enabled' => false,
        'key' => 'wbraid', // can be a string or a regular expression enclosed in / /
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => ''
    ],
    [
        'type' => 'url_param', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(rdn|sku)/i', // can be a string or a regular expression enclosed in / /
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*1234567890.*)/'
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(Accept-Language)/i', // can be a string or a regular expression enclosed in / /
        'safe_if_absent' => true,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/^\*$/'
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => 'X-Chrome-Connected', // The specific key within the JSON object in the 'headers' column
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*enable_account_consistency.*|.*consistency.*)/'
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(Device-Memory|device-memory)/i', // The specific key within the JSON object in the 'headers' column
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => true,
        'filter_type' => 'none', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => ''
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(Downlink|Rtt)/i', // The specific key within the JSON object in the 'headers' column
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => true,
        'filter_type' => 'none', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => ''
    ],
    [
        'type' => 'url_param', // 'header' or 'url_param'
        'enabled' => false,
        'key' => 'xid', // can be a string or a regular expression enclosed in / /
        'safe_if_absent' => true,
        'safe_if_empty' => true,
        'safe_if_present' => false,
        'filter_type' => 'whitelist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '###XID###',
        'blacklist' => ''
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(Cache-Control|cache-control)/i', // The specific key within the JSON object in the 'headers' column
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*max-age.*)|(.*no-cache.*)/'
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(Accept-Language)/i', // Filtra pelo cabeçalho Accept-Language
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' ou 'blacklist'
        'whitelist' => '',
        'blacklist' => '/^[a-zA-Z*-]{1,2}$/i' // Bloqueia se houver apenas 1 ou 2 caracteres
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(Accept-Language)/i', // can be a string or a regular expression enclosed in / /
        'safe_if_absent' => true,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/\*/'
    ],
    [
        'type' => 'url_param', // 'header' or 'url_param'
        'enabled' => false,
        'key' => 'gclid', // can be a string or a regular expression enclosed in / /
        'safe_if_absent' => true,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*EAI.*)|(.*EAIaIQobCh.*)/'
    ],
    [
        'type' => 'header',
        'enabled' => true,
        'key' => '/(ip-api-as|Ip-Api-As)/',
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist',
        'whitelist' => '',
        'blacklist' => '/(.*AS15169.*)|(.*AS19527.*)|(.*AS36492.*)|(.*AS41264.*)|(.*AS396982.*)|(.*AS45566.*)|(.*AS36040.*)|(.*AS43515.*)|(.*AS139190.*)|(.*AS142745.*)|(.*AS16509.*)|(.*AS8075.*)|(.*AS14061.*)|(.*AS40676.*)|(.*AS8100.*)|(.*AS46844.*)|(.*AS53667.*)|(.*AS16276.*)|(.*AS63949.*)|(.*AS20940.*)|(.*AS6185.*)|(.*AS394695.*)|(.*AS13335.*)|(.*AS18978.*)|(.*AS2906.*)|(.*AS20473.*)|(.*AS21859.*)|(.*AS263411.*)|(.*AS263412.*)|(.*AS29789.*)|(.*AS53006.*)|(.*AS135295.*)|(.*AS140202.*)|(.*AS140203.*)|(.*AS140204.*)|(.*AS140205.*)|(.*AS140206.*)|(.*AS140207.*)|(.*AS140208.*)|(.*AS140209.*)|(.*AS140210.*)|(.*AS140211.*)|(.*AS140212.*)/'
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(Accept-Language)/i', // can be a string or a regular expression enclosed in / /
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/^[a-z]{2}$/i' // Esta regex corresponde exatamente a 2 letras (case insens
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(Accept-Language)/i', // can be a string or a regular expression enclosed in / /
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/[a-z]{2}-[a-z]{2}/' // Captura códigos de idioma com todas as letras minúsculas
    ],
    [
        'type' => 'url_param', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(ref_id|rid|gclid)/i', // can be a string or a regular expression enclosed in / /
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/.*gclid.*/'
    ],
    [
        'type' => 'header', // analisa os headers HTTP
        'enabled' => true,
        'key' => '/(Save-Data)/i', // verifica o header 'Sec-Ch-Ua'
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => true,
        'filter_type' => 'blacklist',
        'whitelist' => '',
        'blacklist' => '/on/i'
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(Cookie|cookie)/i', // The specific key within the JSON object in the 'headers' column
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*ar_debug.*)|(.*debug.*)|(.*debugger.*)/'
    ],
    [
        'type' => 'header', // analisa os headers HTTP
        'enabled' => true,
        'key' => '/(cf_chl_rc_m)/i', // verifica o header 'cf_chl_rc_m'
        'safe_if_absent' => false, // SE não existir, é seguro
        'safe_if_empty' => false, // SE existir mas vazio, é inseguro
        'safe_if_present' => true, // SE existir, é inseguro
        'filter_type' => 'none',
        'whitelist' => '',
        'blacklist' => '' // qualquer valor
    ],
    [
        'type' => 'header', // analisa os headers HTTP
        'enabled' => true,
        'key' => '/Sec-Ch-Ua-Model/i', // verifica o header 'Sec-Ch-Ua'
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => true,
        'filter_type' => 'none',
        'whitelist' => '',
        'blacklist' => ''
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => false,
        'key' => '/(referer|referrer|Referer|Referrer|Origin)/i', // can be a string or a regular expression enclosed in / /
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'none', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => ''
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(X-Requested-With)/i', // can be a string or a regular expression enclosed in / /
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*proxy.*)|(https:\/\/youtube\.com\/mobile)|(https:\/\/securepubads\.g\.doubleclick\.net)|(.*securepubads.*)|(.*googleads\.g\.doubleclick\.net\/pagead\/ads.*)|(.*imasdk\.googleapis\.com.*)|(.*imasdk.*)|(.*pagead.*)|(.*www\.ads\.google\.com.*)|(.*duckduckgo.*)|(.*secure.*)|(https:\/\/tdsf\.doubleclick\.net)|(.*shorts.*)|(.*watch\?v.*)|(.*sbrowser.*)|(com\.google\.android\.apps\..*)|(android-app:\/\/com\.google\.android\.googlequicksearchbox\/https\/www\.google\.com)|(.*lookup.*)|(.*html.*)|(https:\/\/googleads\.g\.doubleclick\.net\/pagead\/html\/r[0-9]+\/r[0-9]+\/zrt_lookup\.html.*)|(com\.google\.android\.apps\.magazines)|(.*googleadservices.*)|(.*syndicatedsearch.*)|(.*bing.*)|(.*block.*)|(.*guard.*)|(com\.go\.browser)|(com\.opera\.gx)|(.*browser.*)|(.*bigfont.*)|(.*font.*)/'
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(Sec-Fetch-Dest)/i', // can be a string or a regular expression enclosed in / /
        'safe_if_absent' => true,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'none', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => ''
    ],
    [
        'type' => 'header', // analisa os headers HTTP
        'enabled' => false,
        'key' => '/(Dnt)/i', // O mesmo que contem o Not a Brand, ou seja se nao contem Not a brand
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => true,
        'filter_type' => 'none',
        'whitelist' => '',
        'blacklist' => ''
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => false,
        'key' => '/(referer|referrer|Referer|Referrer|X-Requested-With|Origin)/i', // Continua com a insensibilidade a maiúsculas/minúsculas
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'whitelist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '/(.*wbraid.*)|(.*gbraid.*)/',
        'blacklist' => ''
    ],
    [
        'type' => 'header',
        'enabled' => false,
        'key' => '/(User-Agent)/i',
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*SamsungBrowser.*)|(.*OPR.*)|(.*Firefox.*)|(.*Edg.*)|(.*pageburst.*)|(Chrome\/(?!\d+\.0\.0\.0))|(.*GSA.*)|(.*ddg.*)|(.*CriOS.*)/'
    ],
    [
        'type' => 'header',
        'enabled' => true,
        'key' => '/(User-Agent)/i',
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*pageburst.*)|(.*OPPO.*)|(.*ddg.*)/'
    ],
    [
        'type' => 'header',
        'enabled' => false,
        'key' => '/(X-Real-Ip)/i',
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/^[\s\S]{0,3}[a-f]/i'
    ],
    [
        'type' => 'header', // analisa os headers HTTP
        'enabled' => false,
        'key' => '/(Sec-Ch-Ua)/i', // verifica o header 'Sec-Ch-Ua'
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist',
        'whitelist' => '',
        'blacklist' => '/(\sNot.?A.?Brand|\.Not.?A.?Brand|Not\(A:Brand|Not:A-Brand|Not\.A\/Brand|Not A\(?.?Brand)/i'
    ],
    [
        'type' => 'header', // analisa os headers HTTP
        'enabled' => false,
        'key' => '/(Priority)/i', // verifica o header 'Priority'
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'none',
        'whitelist' => '',
        'blacklist' => ''
    ],
    [
        'type' => 'url_param', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(cr|utm_term)/i', // can be a string or a regular expression enclosed in / /
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/^0$/'
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(X-Requested-With)/i', // can be a string or a regular expression enclosed in / /
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*instabridge.*)|(.*easy\.launcher.*)|(.*opera.*)|(.*unblock.*)|(.*daum.*)|(.*bing.*)/'
    ],
    [
        'type' => 'header',
        'enabled' => false,
        'key' => '/(User-Agent)/i',
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*GSA.*)|(.*ddg.*)/'
    ],
    [
        'type' => 'combined', // 'header' or 'url_param'
        'enabled' => true,
        'subrules' => [
          [
              'type' => 'url_param', // 'header' or 'url_param'
              'enabled' => true,
              'key' => '/(mtx|utm_content)/', // can be a string or a regular expression enclosed in / /
              'safe_if_absent' => false,
              'safe_if_empty' => false,
              'safe_if_present' => false,
              'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
              'whitelist' => '',
              'blacklist' => '/(.*b.*)/'
          ],
            [
                'type' => 'url_param', // 'header' or 'url_param'
                'enabled' => true,
                'key' => '/(kw|utm_medium)/', // can be a string or a regular expression enclosed in / /
                'safe_if_absent' => false,
                'safe_if_empty' => true,
                'safe_if_present' => false,
                'filter_type' => 'none', // 'none', 'whitelist' or 'blacklist'
                'whitelist' => '',
                'blacklist' => ''
            ],
        ],
    ],
    [
        'type' => 'combined', // 'header' or 'url_param'
        'enabled' => false,
        'subrules' => [
          [
              'type' => 'header', // 'header' or 'url_param'
              'enabled' => true,
              'key' => '/(User-Agent)/i', // rule for filtering User-Agent and user-agent
              'safe_if_absent' => false,
              'safe_if_empty' => false,
              'safe_if_present' => false,
              'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
              'whitelist' => '',
              'blacklist' => '/(.*samsung.*)|(.*iphone.*)/i'
          ],
          [
              'type' => 'header', // 'header' or 'url_param'
              'enabled' => true,
              'key' => '/(User-Agent)/i', // rule for filtering User-Agent and user-agent
              'safe_if_absent' => false,
              'safe_if_empty' => false,
              'safe_if_present' => false,
              'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
              'whitelist' => '',
              'blacklist' => '/(.*18_6_2.*)|(.*Chrome\/130\.0\.0\.0.*)/i'
          ],
        ],
    ],
    [
        'type' => 'combined', // 'header' or 'url_param'
        'enabled' => false,
        'subrules' => [
          [
              'type' => 'header', // 'header' or 'url_param'
              'enabled' => true,
              'key' => '/(User-Agent)/i', // rule for filtering User-Agent and user-agent
              'safe_if_absent' => false,
              'safe_if_empty' => false,
              'safe_if_present' => false,
              'filter_type' => 'whitelist', // 'none', 'whitelist' or 'blacklist'
              'whitelist' => '',
              'blacklist' => '/(.*iPhone.*)/i'
          ],
          [
              'type' => 'url_param', // 'header' or 'url_param'
              'enabled' => true,
              'key' => 'gclid', // can be a string or a regular expression enclosed in / /
              'safe_if_absent' => false,
              'safe_if_empty' => false,
              'safe_if_present' => false,
              'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
              'whitelist' => '',
              'blacklist' => '/.*EAI.*/'
          ],
        ],
    ],
    [
        'type' => 'combined', // nao tem nem underscore nem hyphen
        'enabled' => false,
        'subrules' => [
          [
              'type' => 'url_param',
              'enabled' => true,
              'key' => 'gclid',
              'safe_if_absent' => false,
              'safe_if_empty' => false,
              'safe_if_present' => false,
              'filter_type' => 'whitelist',
              'whitelist' => '/^[^_-]*$/',
              'blacklist' => ''
          ],
          [
              'type' => 'url_param',
              'enabled' => true,
              'key' => 'wbraid',
              'safe_if_absent' => false,
              'safe_if_empty' => false,
              'safe_if_present' => false,
              'filter_type' => 'whitelist',
              'whitelist' => '/^[^_-]*$/',
              'blacklist' => ''
          ],
        ],
    ],
    [
        'type' => 'combined', // nao tem nem underscore nem hyphen
        'enabled' => false,
        'subrules' => [
          [
              'type' => 'url_param', // 'header' or 'url_param'
              'enabled' => true,
              'key' => 'gclid', // can be a string or a regular expression enclosed in / /
              'safe_if_absent' => false,
              'safe_if_empty' => false,
              'safe_if_present' => false,
              'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
              'whitelist' => '',
              'blacklist' => '/.*EAI.*/'
          ],
          [
              'type' => 'url_param',
              'enabled' => true,
              'key' => 'wbraid',
              'safe_if_absent' => false,
              'safe_if_empty' => false,
              'safe_if_present' => false,
              'filter_type' => 'blacklist',
              'whitelist' => '',
              'blacklist' => '/./'
          ],
        ],
    ],
    [
        'type' => 'header', // analisa os headers HTTP
        'enabled' => true,
        'key' => '/(Priority)/i', // verifica o header 'Priority'
        'safe_if_absent' => false,
        'safe_if_empty' => false,
        'safe_if_present' => false,
        'filter_type' => 'blacklist',
        'whitelist' => '',
        'blacklist' => '/(u=3)|(u=4)|(u=5)|(u=6)|(u=7)/i'
    ],
    [
        'type' => 'combined', // 'header' or 'url_param'
        'enabled' => true,
        'subrules' => [
            [
                'type' => 'header', // 'header' or 'url_param'
                'enabled' => true,
                'key' => '/(User-Agent)/i', // rule for filtering User-Agent and user-agent
                'safe_if_absent' => false,
                'safe_if_empty' => false,
                'safe_if_present' => false,
                'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
                'whitelist' => '',
                'blacklist' => '/(.*windows.*)/i'
            ],
            [
                'type' => 'header', // analisa os headers HTTP
                'enabled' => true,
                'key' => '/(Priority)/i', // verifica o header 'Priority'
                'safe_if_absent' => true,
                'safe_if_empty' => false,
                'safe_if_present' => false,
                'filter_type' => 'none',
                'whitelist' => '',
                'blacklist' => 'i'
            ],
        ],
    ],
];

if (defined('CUSTOM_PARAMS')){
    $custom_params = json_decode(CUSTOM_PARAMS, true);
    if (is_array($custom_params)) {
        foreach ($custom_params as $param) {
            if (isset($param['key']) && isset($param['value'])) {
                $rules[] = [
                    'type' => 'url_param',
                    'enabled' => true,
                    'key' => $param["key"],
                    'safe_if_absent' => true,
                    'safe_if_empty' => true,
                    'safe_if_present' => false,
                    'filter_type' => 'whitelist',
                    'whitelist' => $param["value"],
                    'blacklist' => ''
                ];
            }
        }
    }
}

$bot_rules = [
    [
        'type' => 'metadata', // 'header', 'url_param', metadata, 'var'
        'enabled' => true,
        'key' => 'device',
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => 4, // 1: desktop; 2: mobile; 3: tablet;  4: 'unknown_device'
    ],
    [
        'type' => 'metadata', // 'header', 'url_param', metadata, 'var'
        'enabled' => false,
        'key' => 'is_ip_api_proxy',
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => 1
    ],
    [
        'type' => 'var', // 'header', 'url_param', metadata, 'var'
        'enabled' => true,
        'key' => 'request',
        'bot_if_absent' => false,
        'bot_if_empty' => true,
        'bot_if_present' => false,
        'filter_type' => 'none', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => ''
    ],
    [
        'type' => 'var',
        'enabled' => true,
        'key' => 'ip', // can be a string or a regular expression enclosed in / /
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/.*face:b00c.*/'
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/^user-agent$/i', // rule for filtering User-Agent and user-agent
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*google.*)|(.*support.*)|(.*Trident.*)|(.*HeadlessChrome.*)|(.*googleweblight.*)|(.*instabridge.*)|(.*trailer.*)/i'
    ],
    [
        'type' => 'var', // 'header' or 'url_param'
        'enabled' => true,
        'key' => 'ip_api_isp', // can be a string or a regular expression enclosed in / /
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*google.*)|(.*cache.*)|(.*proxy.*)|(.*youtube.*)|(.*datacenter.*)|(.*unknown.*)|(.*digitalocean.*)|(.*pubmatic.*)|(.*vultr.*)|(.*xlhdns.*)|(.*addr.*)|(.*amazon.*)|(.*akamai.*)|(.*boost mobile.*)|(.*packethub.*)|(.*clouvider.*)|(.*isomedia.*)|(.*hawaiian.*)|(.*data.*)|(.*level 3.*)|(.*university.*)/'
    ],
    [
        'type' => 'metadata', // 'header' or 'url_param'
        'enabled' => false,
        'key' => 'is_ip_api_proxy', // can be a string or a regular expression enclosed in / /
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => 1
    ],
    [
        'type' => 'var', // 'header' or 'url_param'
        'enabled' => true,
        'key' => 'ip_api_reverse', // can be a string or a regular expression enclosed in / /
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*google.*)|(.*cache.*)|(.*proxy.*)|(.*youtube.*)|(.*datacenter.*)|(.*unknown.*)|(.*digitalocean.*)|(.*pubmatic.*)|(.*vultr.*)|(.*xlhdns.*)|(.*addr.*)|(.*amazon.*)|(.*akamai.*)|(.*data.*)/'
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => false,
        'key' => '/(ip-api-hosting|Ip-Api-Hosting)/', // can be a string or a regular expression enclosed in / /
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => 1
    ],
    [
        'type' => 'url_param', // 'header' or 'url_param'
        'enabled' => false,
        'key' => 'domain', // can be a string or a regular expression enclosed in / /
        'bot_if_absent' => true,
        'bot_if_empty' => true,
        'bot_if_present' => false,
        'filter_type' => 'whitelist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '/.*domain.*/',
        'blacklist' => ''
    ],
    [
        'type' => 'url_param', // 'header' or 'url_param'
        'enabled' => true,
        'key' => 'gclid', // can be a string or a regular expression enclosed in / /
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*gclid.*)/'
    ],
    [
        'type' => 'url_param', // 'header' or 'url_param'
        'enabled' => false,
        'key' => 'wbraid', // can be a string or a regular expression enclosed in / /
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => ''
    ],
    [
        'type' => 'url_param', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(rdn|sku)/i', // can be a string or a regular expression enclosed in / /
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*1234567890.*)/'
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(Accept-Language)/i', // can be a string or a regular expression enclosed in / /
        'bot_if_absent' => true,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'none', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/^\*$/'
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => 'X-Chrome-Connected', // The specific key within the JSON object in the 'headers' column
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*enable_account_consistency.*|.*consistency.*)/'
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(Device-Memory|device-memory)/i', // The specific key within the JSON object in the 'headers' column
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => true,
        'filter_type' => 'none', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => ''
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(Downlink|Rtt)/i', // The specific key within the JSON object in the 'headers' column
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => true,
        'filter_type' => 'none', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => ''
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(Cookie|cookie)/i', // The specific key within the JSON object in the 'headers' column
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*ar_debug.*)|(.*debug.*)|(.*debugger.*)/'
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(Cache-Control|cache-control)/i', // The specific key within the JSON object in the 'headers' column
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*max-age.*)|(.*no-cache.*)/'
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(Accept-Language)/i', // Filtra pelo cabeçalho Accept-Language
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' ou 'blacklist'
        'whitelist' => '',
        'blacklist' => '/^[a-zA-Z*-]{1,2}$/i' // Bloqueia se houver apenas 1 ou 2 caracteres
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(Accept-Language)/i', // can be a string or a regular expression enclosed in / /
        'bot_if_absent' => true,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/\*/'
    ],
    [
        'type' => 'header',
        'enabled' => true,
        'key' => '/(ip-api-as|Ip-Api-As)/',
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist',
        'whitelist' => '',
        'blacklist' => '/(.*AS15169.*)|(.*AS19527.*)|(.*AS36492.*)|(.*AS41264.*)|(.*AS396982.*)|(.*AS45566.*)|(.*AS36040.*)|(.*AS43515.*)|(.*AS139190.*)|(.*AS142745.*)|(.*AS16509.*)|(.*AS8075.*)|(.*AS14061.*)|(.*AS40676.*)|(.*AS8100.*)|(.*AS46844.*)|(.*AS53667.*)|(.*AS16276.*)|(.*AS63949.*)|(.*AS20940.*)|(.*AS6185.*)|(.*AS394695.*)|(.*AS13335.*)|(.*AS18978.*)|(.*AS2906.*)|(.*AS20473.*)|(.*AS21859.*)|(.*AS263411.*)|(.*AS263412.*)|(.*AS29789.*)|(.*AS53006.*)|(.*AS135295.*)|(.*AS140202.*)|(.*AS140203.*)|(.*AS140204.*)|(.*AS140205.*)|(.*AS140206.*)|(.*AS140207.*)|(.*AS140208.*)|(.*AS140209.*)|(.*AS140210.*)|(.*AS140211.*)|(.*AS140212.*)/'
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(Accept-Language)/i', // can be a string or a regular expression enclosed in / /
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/^[a-z]{2}$/i' // Esta regex corresponde exatamente a 2 letras (case insens
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(Accept-Language)/i', // can be a string or a regular expression enclosed in / /
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/[a-z]{2}-[a-z]{2}/' // Captura códigos de idioma com todas as letras minúsculas
    ],
    [
        'type' => 'url_param', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(ref_id|rid|gclid)/i', // can be a string or a regular expression enclosed in / /
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/.*gclid.*/'
    ],
    [
        'type' => 'header', // analisa os headers HTTP
        'enabled' => true,
        'key' => '/(Save-Data)/i', // verifica o header 'Sec-Ch-Ua'
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => true,
        'filter_type' => 'blacklist',
        'whitelist' => '',
        'blacklist' => '/on/i'
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(Cookie|cookie)/i', // The specific key within the JSON object in the 'headers' column
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*ar_debug.*)|(.*debug.*)|(.*debugger.*)/'
    ],
    [
        'type' => 'header', // analisa os headers HTTP
        'enabled' => true,
        'key' => '/(cf_chl_rc_m)/i', // verifica o header 'cf_chl_rc_m'
        'bot_if_absent' => false, // SE não existir, é seguro
        'bot_if_empty' => false, // SE existir mas vazio, é inseguro
        'bot_if_present' => true, // SE existir, é inseguro
        'filter_type' => 'none',
        'whitelist' => '',
        'blacklist' => '' // qualquer valor
    ],
    [
        'type' => 'header', // analisa os headers HTTP
        'enabled' => true,
        'key' => '/Sec-Ch-Ua-Model/i', // verifica o header 'Sec-Ch-Ua'
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => true,
        'filter_type' => 'none',
        'whitelist' => '',
        'blacklist' => ''
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => false,
        'key' => '/(referer|referrer|Referer|Referrer|Origin)/i', // can be a string or a regular expression enclosed in / /
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'none', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => ''
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(X-Requested-With)/i', // can be a string or a regular expression enclosed in / /
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*proxy.*)|(https:\/\/youtube\.com\/mobile)|(https:\/\/securepubads\.g\.doubleclick\.net)|(.*securepubads.*)|(.*googleads\.g\.doubleclick\.net\/pagead\/ads.*)|(.*imasdk\.googleapis\.com.*)|(.*imasdk.*)|(.*pagead.*)|(.*www\.ads\.google\.com.*)|(.*duckduckgo.*)|(.*secure.*)|(https:\/\/tdsf\.doubleclick\.net)|(.*shorts.*)|(.*watch\?v.*)|(.*sbrowser.*)|(com\.google\.android\.apps\..*)|(android-app:\/\/com\.google\.android\.googlequicksearchbox\/https\/www\.google\.com)|(.*lookup.*)|(.*html.*)|(https:\/\/googleads\.g\.doubleclick\.net\/pagead\/html\/r[0-9]+\/r[0-9]+\/zrt_lookup\.html.*)|(com\.google\.android\.apps\.magazines)|(.*googleadservices.*)|(.*syndicatedsearch.*)|(.*bing.*)|(.*block.*)|(.*guard.*)|(com\.go\.browser)|(com\.opera\.gx)|(.*browser.*)|(.*bigfont.*)|(.*font.*)/'
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(Sec-Fetch-Dest)/i', // can be a string or a regular expression enclosed in / /
        'bot_if_absent' => true,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'none', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => ''
    ],
    [
        'type' => 'header', // analisa os headers HTTP
        'enabled' => false,
        'key' => '/(Dnt)/i', // O mesmo que contem o Not a Brand, ou seja se nao contem Not a brand
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => true,
        'filter_type' => 'none',
        'whitelist' => '',
        'blacklist' => ''
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => false,
        'key' => '/(referer|referrer|Referer|Referrer|X-Requested-With|Origin)/i', // Continua com a insensibilidade a maiúsculas/minúsculas
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'whitelist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '/(.*wbraid.*)|(.*gbraid.*)/',
        'blacklist' => ''
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => false,
        'key' => '/(referer|referrer|Referer|Referrer|Origin)/i', // Continua com a insensibilidade a maiúsculas/minúsculas
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'whitelist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '/(.*gad_source.*)/',
        'blacklist' => ''
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => false,
        'key' => '/(X-Requested-With)/i', // can be a string or a regular expression enclosed in / /
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => true,
        'filter_type' => 'none', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => ''
    ],
    [
        'type' => 'header',
        'enabled' => false,
        'key' => '/(User-Agent)/i',
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*OPR.*)|(.*Firefox.*)|(.*Edg.*)|(.*pageburst.*)|(Chrome\/(?!\d+\.0\.0\.0))|(.*GSA.*)|(.*ddg.*)|(.*CriOS.*)/'
    ],
    [
        'type' => 'header',
        'enabled' => true,
        'key' => '/(User-Agent)/i',
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*pageburst.*)|(.*OPPO.*)|(.*ddg.*)/'
    ],
    [
        'type' => 'header',
        'enabled' => true,
        'key' => '/(X-Real-Ip)/i',
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/^[\s\S]{0,3}[a-f]/i'
    ],
    [
        'type' => 'header', // analisa os headers HTTP
        'enabled' => false,
        'key' => '/(Sec-Ch-Ua)/i', // verifica o header 'Sec-Ch-Ua'
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist',
        'whitelist' => '',
        'blacklist' => '/(\sNot.?A.?Brand|\.Not.?A.?Brand|Not\(A:Brand|Not:A-Brand|Not\.A\/Brand|Not A\(?.?Brand)/i'
    ],
    [
        'type' => 'header', // analisa os headers HTTP
        'enabled' => false,
        'key' => '/(Priority)/i', // verifica o header 'Priority'
        'bot_if_absent' => true,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'none',
        'whitelist' => '',
        'blacklist' => ''
    ],
    [
        'type' => 'header', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(X-Requested-With)/i', // can be a string or a regular expression enclosed in / /
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*instabridge.*)|(.*easy\.launcher.*)|(.*opera.*)|(.*unblock.*)|(.*daum.*)|(.*bing.*)/'
    ],
    [
        'type' => 'url_param', // 'header' or 'url_param'
        'enabled' => true,
        'key' => '/(cr|utm_term)/i', // can be a string or a regular expression enclosed in / /
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/^0$/'
    ],
    [
        'type' => 'header',
        'enabled' => false,
        'key' => '/(User-Agent)/i',
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*GSA.*)|(.*ddg.*)/'
    ],
    [
        'type' => 'combined', // 'header' or 'url_param'
        'enabled' => true,
        'subrules' => [
            [
                'type' => 'url_param', // 'header' or 'url_param'
                'enabled' => true,
                'key' => '/(mtx|utm_content)/', // can be a string or a regular expression enclosed in / /
                'bot_if_absent' => false,
                'bot_if_empty' => false,
                'bot_if_present' => false,
                'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
                'whitelist' => '',
                'blacklist' => '/(.*b.*)/'
            ],
            [
                'type' => 'url_param', // 'header' or 'url_param'
                'enabled' => true,
                'key' => '/(kw|utm_medium)/', // can be a string or a regular expression enclosed in / /
                'bot_if_absent' => false,
                'bot_if_empty' => true,
                'bot_if_present' => false,
                'filter_type' => 'none', // 'none', 'whitelist' or 'blacklist'
                'whitelist' => '',
                'blacklist' => ''
            ],
        ],
    ],
    [
        'type' => 'combined', // 'header' or 'url_param'
        'enabled' => false,
        'subrules' => [
            [
                'type' => 'header', // 'header' or 'url_param'
                'enabled' => true,
                'key' => '/(User-Agent)/i', // rule for filtering User-Agent and user-agent
                'bot_if_absent' => false,
                'bot_if_empty' => false,
                'bot_if_present' => false,
                'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
                'whitelist' => '',
                'blacklist' => '/(.*samsung.*)|(.*iphone.*)/i'
            ],
            [
                'type' => 'header', // 'header' or 'url_param'
                'enabled' => true,
                'key' => '/(User-Agent)/i', // rule for filtering User-Agent and user-agent
                'bot_if_absent' => false,
                'bot_if_empty' => false,
                'bot_if_present' => false,
                'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
                'whitelist' => '',
                'blacklist' => '/(.*18_6_2.*)|(.*Chrome\/130\.0\.0\.0.*)/i'
            ],
        ],
    ],
    [
        'type' => 'combined', // 'header' or 'url_param'
        'enabled' => false,
        'subrules' => [
            [
                'type' => 'header', // 'header' or 'url_param'
                'enabled' => true,
                'key' => '/(User-Agent)/i', // rule for filtering User-Agent and user-agent
                'bot_if_absent' => false,
                'bot_if_empty' => false,
                'bot_if_present' => false,
                'filter_type' => 'whitelist', // 'none', 'whitelist' or 'blacklist'
                'whitelist' => '',
                'blacklist' => '/(.*iPhone.*)/i'
            ],
            [
                'type' => 'url_param', // 'header' or 'url_param'
                'enabled' => true,
                'key' => 'gclid', // can be a string or a regular expression enclosed in / /
                'bot_if_absent' => false,
                'bot_if_empty' => false,
                'bot_if_present' => false,
                'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
                'whitelist' => '',
                'blacklist' => '/.*EAI.*/'
            ],
        ],
    ],

    [
        'type' => 'combined', // 'header' or 'url_param'
        'enabled' => false,
        'subrules' => [
            [
                'type' => 'url_param',
                'enabled' => true,
                'key' => 'gclid',
                'bot_if_absent' => false,
                'bot_if_empty' => false,
                'bot_if_present' => false,
                'filter_type' => 'whitelist',
                'whitelist' => '/^[^_-]*$/',
                'blacklist' => ''
            ],
            [
                'type' => 'url_param',
                'enabled' => true,
                'key' => 'wbraid',
                'bot_if_absent' => false,
                'bot_if_empty' => false,
                'bot_if_present' => false,
                'filter_type' => 'whitelist',
                'whitelist' => '/^[^_-]*$/',
                'blacklist' => ''
            ],
        ],
    ],
    [
        'type' => 'combined', // nao tem nem underscore nem hyphen
        'enabled' => false,
        'subrules' => [
            [
                'type' => 'url_param', // 'header' or 'url_param'
                'enabled' => true,
                'key' => 'gclid', // can be a string or a regular expression enclosed in / /
                'bot_if_absent' => false,
                'bot_if_empty' => false,
                'bot_if_present' => false,
                'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
                'whitelist' => '',
                'blacklist' => '/.*EAI.*/'
            ],
            [
                'type' => 'url_param',
                'enabled' => true,
                'key' => 'wbraid',
                'bot_if_absent' => false,
                'bot_if_empty' => false,
                'bot_if_present' => false,
                'filter_type' => 'blacklist',
                'whitelist' => '',
                'blacklist' => '/./'
            ],
        ],
    ],
    [
        'type' => 'header',
        'enabled' => true,
        'device' => '/(desktop)/',
        'key' => '/(User-Agent)/i',
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
        'whitelist' => '',
        'blacklist' => '/(.*pageburst.*)|(.*ddg.*)|(.*Windows NT 6\.1.*)|(.*Windows NT 6\.3.*)|(.*Android-Gmail.*)/'
    ],
    [
        'type' => 'header', // analisa os headers HTTP
        'enabled' => true,
        'key' => '/(Priority)/i', // verifica o header 'Priority'
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist',
        'whitelist' => '',
        'blacklist' => '/(u=3)|(u=4)|(u=5)|(u=6)|(u=7)/i'
    ],
    [
        'type' => 'combined', // 'header' or 'url_param'
        'enabled' => true,
        'subrules' => [
            [
                'type' => 'header', // 'header' or 'url_param'
                'enabled' => true,
                'key' => '/(User-Agent)/i', // rule for filtering User-Agent and user-agent
                'bot_if_absent' => false,
                'bot_if_empty' => false,
                'bot_if_present' => false,
                'filter_type' => 'blacklist', // 'none', 'whitelist' or 'blacklist'
                'whitelist' => '',
                'blacklist' => '/(.*windows.*)/i'
            ],
            [
                'type' => 'header', // analisa os headers HTTP
                'enabled' => true,
                'key' => '/(Priority)/i', // verifica o header 'Priority'
                'bot_if_absent' => true,
                'bot_if_empty' => false,
                'bot_if_present' => false,
                'filter_type' => 'none',
                'whitelist' => '',
                'blacklist' => 'i'
            ],
        ],
    ],
    [
        'type' => 'header',
        'enabled' => true,
        'key' => '/(X-Asn)/i',
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist',
        'whitelist' => '',
        'blacklist' => '/^(1138699|15169|16509|14618|396982|36040|22616|54634|36692|174|7029|212238|14061|9009|16276|132203|8075|32934|135377|209366|213230|46475|200478|13238|714|210743|22075|136907|2200|24940|23033|140577|4837|43037|32475|13335|51167|23724|48282|7941|29695|13414|2119|50300|3209|27281|153568|199739|216341|3257|39264|50340|211860|6697|7552|16010|209290|35048|25513|31520|12389|136958|50304|50113|34665|202425|21837)$/' // Bloqueia exatamente 2 caracteres minúsculos
    ],
    [
        'type' => 'header',
        'enabled' => true,
        'key' => '/(Ip-Api-As|Ip-Api-Asname|Ip-Api-Asname)/i',
        'bot_if_absent' => false,
        'bot_if_empty' => false,
        'bot_if_present' => false,
        'filter_type' => 'blacklist',
        'whitelist' => '',
        'blacklist' => '/(.*google.*)/i' // Bloqueia exatamente 2 caracteres minúsculos
    ],
];

$debug = false;


include '../twr/cloaker/l34t0734m0/cloaker302-proxy.php';
?>