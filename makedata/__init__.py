import makedata.getBreastCancerData
import makedata.getCryotherapyData
import makedata.getDivorceData
import makedata.getWineData
import makedata.getForestFireData
import makedata.getAuditRiskData
import makedata.getR2Data
import makedata.getHabermanData
import makedata.getHeartFailureData
import makedata.getHillValleyData
import makedata.getLiverPatientData
import makedata.getIonosphereData


def load_data(name="breastCancer"):
    if name == "breastCancer":
        return makedata.getBreastCancerData.load_data()
    elif name == "cryotherapy":
        return makedata.getCryotherapyData.load_data()
    elif name == "divorce":
        return makedata.getDivorceData.load_data()
    elif name == "wine":
        return makedata.getWineData.load_data()
    elif name == "forestFire":
        return makedata.getForestFireData.load_data()
    elif name == "auditRisk":
        return makedata.getAuditRiskData.load_data()
    elif name == "r2":
        return makedata.getR2Data.load_data()
    elif name == "haberman":
        return makedata.getHabermanData.load_data()
    elif name == "heartFailure":
        return makedata.getHeartFailureData.load_data()
    elif name == "hillValley":
        return makedata.getHillValleyData.load_data()
    elif name == "liverPatient":
        return makedata.getLiverPatientData.load_data()
    elif name == "ionosphere":
        return makedata.getIonosphereData.load_data()

    return None
