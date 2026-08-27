from __future__ import annotations

_MLP_SELECTOR_ALIASES = {
    ("pyg", "2cb02ce31438f6beb9f7024e17a78fc6fd5a917c73d338f7b1a287f9ece61c6e", False): "7f6bdfabbcb80370623999728bee682876c981f4c504c596beb4e0a08f4ea60a",
    ("pyg", "2cb02ce31438f6beb9f7024e17a78fc6fd5a917c73d338f7b1a287f9ece61c6e", True): "7f6bdfabbcb80370623999728bee682876c981f4c504c596beb4e0a08f4ea60a",
    ("ogbl", "a741ea6be4049d53ae65f45963186fadc839de3bad94238592e679be3fd896a8", False): "33ec331bea2d2dbfddac3f75d0b8037ff1e4e617c738808dda7ee74af5300acf",
    ("ogbl", "a741ea6be4049d53ae65f45963186fadc839de3bad94238592e679be3fd896a8", True): "33ec331bea2d2dbfddac3f75d0b8037ff1e4e617c738808dda7ee74af5300acf",
    ("pyg", "de6ca97b8131abaa4038604bfdddd8099e27b7cb229b29d38efedd5ae5634062", False): "7f6bdfabbcb80370623999728bee682876c981f4c504c596beb4e0a08f4ea60a",
    ("pyg", "de6ca97b8131abaa4038604bfdddd8099e27b7cb229b29d38efedd5ae5634062", True): "7f6bdfabbcb80370623999728bee682876c981f4c504c596beb4e0a08f4ea60a",
    ("ogbl", "8507a60128712fa9fb32a9a2bb09ed01a31cc758039ccbe1b051d0b18bcd3037", False): "33ec331bea2d2dbfddac3f75d0b8037ff1e4e617c738808dda7ee74af5300acf",
    ("ogbl", "8507a60128712fa9fb32a9a2bb09ed01a31cc758039ccbe1b051d0b18bcd3037", True): "33ec331bea2d2dbfddac3f75d0b8037ff1e4e617c738808dda7ee74af5300acf",
    ("pyg", "8db97f3154bfc041b0126b909c66ac796c9117f7527d1914da6984c81dd3989a", False): "7f6bdfabbcb80370623999728bee682876c981f4c504c596beb4e0a08f4ea60a",
    ("pyg", "8db97f3154bfc041b0126b909c66ac796c9117f7527d1914da6984c81dd3989a", True): "7f6bdfabbcb80370623999728bee682876c981f4c504c596beb4e0a08f4ea60a",
    ("ogbl", "966af6d6729ee65e67a5ce2f2e0cd5a4f7d9870c99d947d2af5b5ae3275d21c8", False): "33ec331bea2d2dbfddac3f75d0b8037ff1e4e617c738808dda7ee74af5300acf",
    ("ogbl", "966af6d6729ee65e67a5ce2f2e0cd5a4f7d9870c99d947d2af5b5ae3275d21c8", True): "33ec331bea2d2dbfddac3f75d0b8037ff1e4e617c738808dda7ee74af5300acf",
    ("pyg", "c9d98dd643a593eb47007884a8e52854726bb3a279ea001266355aeeadff5c1c", False): "ad423ad82944f13fa8940396e09a21a959ac1e3e6d0ee80ba109388044e91580",
    ("pyg", "c9d98dd643a593eb47007884a8e52854726bb3a279ea001266355aeeadff5c1c", True): "cf3bb9d2469f6fd9b7c59c0170c4e062dc816005c9f79304c13b7883697b3d37",
    ("pyg", "36cc7cc9154c87a8b8826a6c6c05803467245ee67a83f48ab9a328dbebe2cbb4", False): "ad423ad82944f13fa8940396e09a21a959ac1e3e6d0ee80ba109388044e91580",
    ("pyg", "36cc7cc9154c87a8b8826a6c6c05803467245ee67a83f48ab9a328dbebe2cbb4", True): "cf3bb9d2469f6fd9b7c59c0170c4e062dc816005c9f79304c13b7883697b3d37",
    ("pyg", "24cf71bd14bd012d831c1b65f227dfdb8039792526593e4829e454041964c9a8", False): "ad423ad82944f13fa8940396e09a21a959ac1e3e6d0ee80ba109388044e91580",
    ("pyg", "24cf71bd14bd012d831c1b65f227dfdb8039792526593e4829e454041964c9a8", True): "cf3bb9d2469f6fd9b7c59c0170c4e062dc816005c9f79304c13b7883697b3d37",
    ("ogbl", "b39faebd9129b90e53579f10bb5de7d43f4e114e734886f75a339fd82e16d156", False): "b04fcf78022b3b61c6b89c20dca6e083a286c0bcbf40795696d0d25d22579184",
    ("ogbl", "b39faebd9129b90e53579f10bb5de7d43f4e114e734886f75a339fd82e16d156", True): "041ad1134a076219e9a18cae82ccc46a4fb10667286b3b544ad0cce153431539",
    ("ogbl", "719a66d39726cd23f4de63f5fa8b1840a3bf9d33c0bd4d38afc902af00f0ae2b", False): "b04fcf78022b3b61c6b89c20dca6e083a286c0bcbf40795696d0d25d22579184",
    ("ogbl", "719a66d39726cd23f4de63f5fa8b1840a3bf9d33c0bd4d38afc902af00f0ae2b", True): "041ad1134a076219e9a18cae82ccc46a4fb10667286b3b544ad0cce153431539",
    ("ogbl", "efaf6d73b159195f29856737adcd92ef6679c74f4dde8e7728a300e1ad27ad43", False): "b04fcf78022b3b61c6b89c20dca6e083a286c0bcbf40795696d0d25d22579184",
    ("ogbl", "efaf6d73b159195f29856737adcd92ef6679c74f4dde8e7728a300e1ad27ad43", True): "041ad1134a076219e9a18cae82ccc46a4fb10667286b3b544ad0cce153431539",
}
_MLPIP_SELECTOR_ALIASES = {
    ("pyg", "2032dc4984c3b722ae60645c96035526917f17d7aef4bae7a71fdf8df4bd025e"): "bd627ce57e5bcedd2faf37e48ebe8c7a8d7738f593519d6490dca209478d633c",
    ("ogbl", "81629134cf40a7e64598c1a8481c0c73a161ffc7bc208ce8432d4be056da1bd0"): "1c3fc9d0f684a0b371cf8bed53ef029ad31bd8fcdcb1aa9db6256614adb8491b",
    ("pyg", "18a6aeb76c1c053b2a2a7093cc781bc538fcc376cbc1af259333e5c23df68ca0"): "bd627ce57e5bcedd2faf37e48ebe8c7a8d7738f593519d6490dca209478d633c",
    ("ogbl", "d4a356dbc9d49cc09cd00c60206e646ce87018e89bb5484035895104bdd14b71"): "1c3fc9d0f684a0b371cf8bed53ef029ad31bd8fcdcb1aa9db6256614adb8491b",
}
_PYG_HEART_SELECTOR_ALIASES = {
    ("gpu", "60322dee22e3c41a6e5c6e2491356b93bc986b429a631849ebb5a756231748ec"): "036edd1fa827ddd82669a95f176a32f5eca7de3b13472cd12d08dbc57acfd543",
    ("gpu", "7de83f65de45be4924961ec124caba24d268e29ec32e36e3fc194864cec94cff"): "036edd1fa827ddd82669a95f176a32f5eca7de3b13472cd12d08dbc57acfd543"
}
_OGB_HEART_SELECTOR_ALIASES = {
    "3c925c8cb407264f286930d8cb26fe5264222e106d357669e6996945de4c0aa7": "76f1b00beec16152189cf65b1652f04cba49bca6dc6ee071b95c2aede4d6fc84"
}


def relocated_mlp_selector_fingerprint(framework: str, raw_digest: str, *, selector_depth: int | None = None) -> str:
    framework = str(framework).strip().lower()
    framework = "ogbl" if framework == "ogb" else framework
    raw_digest = str(raw_digest)
    return _MLP_SELECTOR_ALIASES.get((framework, raw_digest, selector_depth is not None), raw_digest)


def relocated_mlpip_selector_fingerprint(framework: str, raw_digest: str) -> str:
    framework = str(framework).strip().lower()
    framework = "ogbl" if framework == "ogb" else framework
    raw_digest = str(raw_digest)
    return _MLPIP_SELECTOR_ALIASES.get((framework, raw_digest), raw_digest)


def relocated_pyg_heart_selector_fingerprint(backend: str, raw_digest: str) -> str:
    raw_digest = str(raw_digest)
    return _PYG_HEART_SELECTOR_ALIASES.get((str(backend), raw_digest), raw_digest)


def relocated_ogb_heart_selector_fingerprint(raw_digest: str) -> str:
    raw_digest = str(raw_digest)
    return _OGB_HEART_SELECTOR_ALIASES.get(raw_digest, raw_digest)
