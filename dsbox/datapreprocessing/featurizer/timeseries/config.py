D3M_API_VERSION = '2018.1.5'
VERSION = "0.0.1"
TAG_NAME = "9972d415d1a1c7ad343fd88fea5bb56aa1e71a1d"

REPOSITORY = "https://github.com/luofanghao/featurizer"
PACAKGE_NAME = "dsbox-featurizer"

D3M_PERFORMER_TEAM = 'ISI'

if TAG_NAME:
    PACKAGE_URI = "git+" + REPOSITORY + "@" + TAG_NAME
else:
    PACKAGE_URI = "git+" + REPOSITORY 

PACKAGE_URI = PACKAGE_URI + "#egg=" + PACAKGE_NAME


INSTALLATION_TYPE = 'GIT'
if INSTALLATION_TYPE == 'PYPI':
    INSTALLATION = {
        "type" : "PIP",
        "package": PACAKGE_NAME,
        "version": VERSION
    }
else:
    # INSTALLATION_TYPE == 'GIT'
    INSTALLATION = {
        "type" : "PIP",
        "package_uri": PACKAGE_URI,
    }