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
