

subset7 = [
        "categoryname",
        "overallseverity",
        "score",
        "untrustscore",
        "flowscore",
        "trustscore",
        "enforcementscore",
    ]

subset25 = [
        "correlatedcount",
        "srcip_cd",
        "dstip_cd",
        "srcport_cd",
        "dstport_cd",
        "alerttype_cd",
        "direction_cd",
        "eventname_cd",
        "severity_cd",
        "reportingdevice_cd",
        "devicetype_cd",
        "devicevendor_cd",
        "domain_cd",
        "protocol_cd",
        "username_cd",
        "srcipcategory_cd",
        "dstipcategory_cd",
        "isiptrusted",
        "dstipcategory_dominate",
        "srcipcategory_dominate",
        "dstportcategory_dominate",
        "srcportcategory_dominate",
        "thrcnt_month",
        "thrcnt_week",
        "thrcnt_day",
    ]

tag2selection = {
    7: subset7,
    25: subset25,
    32: [*subset7, *subset25]
}



list_incomplete_columns = [
    "score"
]

_ipcategory_dominate = {
    "INTERNET": 0,
    "PRIV-10": 1,
    "PRIV-CGN": 1,
    "PRIV-172": 1,
    "PRIV-192": 1,
    "BROADCAST": 2,
    "CURR_NET": 2,
    "MULTICAST": 2,
    "LOOPBACK": 2,
    "LINK-LOCAL": 2,
    "BENCH": 2
}

encoding = {
    "categoryname":
        {
            'Attack': 2,
            'Control and Maintain': 2,
            'Reconnaissance': 2,
            'Attack Preparation': 2,
            'Compromise': 1,
            'Exploit': 1,
            'Suspicious Reputation': 0,
            'Suspicious Network Activity': 0,
            'Malicious Activity': 0,
            'To Be Determined': 0,
            'Suspicious Account Activity': 0
        },
    "dstipcategory_dominate": _ipcategory_dominate,
    "srcipcategory_dominate": _ipcategory_dominate,
}

