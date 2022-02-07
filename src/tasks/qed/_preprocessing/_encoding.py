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