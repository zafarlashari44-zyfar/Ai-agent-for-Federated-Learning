from agents.dataset_agent import DatasetAgent
from agents.signal_agent import SignalAgent
from agents.federated_agent import FederatedAgent
from agents.prediction_agent import PredictionAgent
from agents.shap_agent import SHAPAgent
from agents.reasoning_agent import ReasoningAgent
from agents.medical_validation_agent import MedicalValidationAgent
from agents.ehr_agent import EHRAgent
from agents.doctor_agent import DoctorAgent
from agents.patient_agent import PatientAgent
from agents.pdf_agent import PDFAgent
from agents.logger_agent import LoggerAgent
from agents.memory_agent import MemoryAgent
from agents.planner_agent import PlannerAgent


def main():

    logger = LoggerAgent()

    memory = MemoryAgent()

    validator = MedicalValidationAgent()

    dataset_agent = DatasetAgent()

    signal_agent = SignalAgent()

    federated_agent = FederatedAgent()

    prediction_agent = PredictionAgent()

    shap_agent = SHAPAgent()

    reasoning_agent = ReasoningAgent()

    ehr_agent = EHRAgent()

    doctor_agent = DoctorAgent()

    patient_agent = PatientAgent()

    pdf_agent = PDFAgent()

    planner = PlannerAgent(

        dataset_agent,

        signal_agent,

        federated_agent,

        prediction_agent,

        shap_agent,

        reasoning_agent,

        ehr_agent,

        doctor_agent,

        patient_agent,

        pdf_agent,

        memory,

        logger,

    )

    planner.run()


if __name__ == "__main__":

    main()