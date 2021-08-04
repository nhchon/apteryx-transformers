import pandas as pd

MESSAGE = "Hello, world!"

TC_CLASS_MAP = {
    0: "2800 - Semiconductors/Memory, Circuits/Measuring and Testing, Optics/Photocopying, Printing/Measuring and Testing",
    1: "1600 - Biotechnology and Organic Chemistry",
    2: "2600 - Communications",
    3: "3700 - Mechanical Engineering, Manufacturing, Gaming, and Medical Devices/Processes",
    4: "1700 - Chemical and Materials Engineering",
    5: "3600 - Transportation, Construction, Electronic Commerce, Agriculture, National Security and License and Review",
    6: "2100 - Computer Architecture and Software",
    7: "2400 - Networking, Multiplexing, Cable, and Security",
}

TC_COUNT_MAP = {
    2800: 1043341,
    3700: 584685,
    2600: 525022,
    3600: 504068,
    1700: 466615,
    1600: 322352,
    2100: 317652,
    2400: 265921,
    2700: 14765,
    3900: 2212,
    4100: 994,
    2900: 730,
    1800: 71,
    3200: 59,
    2500: 54,
    3300: 48,
    1200: 47,
    3400: 43,
    2300: 38,
    3800: 37,
    3100: 34,
    2000: 28,
    2200: 26,
    3500: 21,
    1100: 20,
    1900: 18,
    1500: 16,
    1000: 13,
    7800: 13,
    7600: 12,
    1300: 12,
    5700: 11,
    3000: 6,
    7700: 4,
    4200: 3,
    5600: 3,
    4700: 2,
    4600: 2,
    5100: 1,
    5400: 1,
    4500: 1,
    9700: 1,
    6500: 1,
    500: 1,
    6700: 1,
    8600: 1,
    4000: 1,
    4800: 1,
    8800: 1,
    1400: 1,
    400: 1,
    7200: 1,
}

TC_TOTAL = 4049012


BLOCKLIST = [
    "fig",
    "fig.",
    "figs",
    "figs.",
    "figure",
    "example",
    "examples",
    "table",
    "tables",
    "jan.",
    "feb.",
    "mar.",
    "apr.",
    "may.",
    "jun.",
    "jul.",
    "aug.",
    "sep.",
    "sept.",
    "oct.",
    "nov.",
    "dec.",
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
]

STRIPLIST = [
    "ie.",
    "i.e.",
    "eg.",
    "e.g.",
]

IDEATION_PROMPT_TEMPLATE = """D1: Aviation
D2: Air systems
Abs: A cabin air system for aviation electronics comprises a compressor, an air conditioner, a mix manifold, and an aviation electronics module. The compressor compresses ambient air, and the air conditioner cools the compressed ambient air to produce cabin air. The mix manifold delivers some of the cabin air to the electronics module.
#########
D1: Memory management
D2: Distributed computing
Abs: Computational storage techniques for distribute computing are disclosed. The computational storage server receives input from multiple clients, which is used by the server when executing one or more computation functions. The computational storage server can aggregate multiple client inputs before applying one or more computation functions. The computational storage server sets up: a first memory area for storing input received from multiple clients; a second memory area designated for storing the computation functions to be executed by the computational storage server using the input data received from the multiple clients; a client specific memory management area for storing metadata related to computations performed by the computational storage server for specific clients; and a persistent storage area for storing checkpoints associated with aggregating computations performed by the computation functions.
#########
D1: {}
D2: {}
Abs:"""
