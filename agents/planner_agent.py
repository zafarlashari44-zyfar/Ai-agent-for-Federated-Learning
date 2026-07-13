class PlannerAgent:

    def __init__(
        self,
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
        memory_agent,
        logger_agent,
    ):

        self.dataset = dataset_agent
        self.signal = signal_agent
        self.federated = federated_agent
        self.prediction = prediction_agent
        self.shap = shap_agent
        self.reasoning = reasoning_agent
        self.ehr = ehr_agent
        self.doctor = doctor_agent
        self.patient = patient_agent
        self.pdf = pdf_agent
        self.memory = memory_agent
        self.logger = logger_agent

    def run(self):

        self.logger.log("AI Agent Started")

        dataset = self.dataset.run()
        self.memory.save("dataset", dataset)

        signals = self.signal.run(dataset)
        self.memory.save("signals", signals)

        self.federated.run()

        prediction = self.prediction.run(signals)
        self.memory.save("prediction", prediction)

        shap = self.shap.run(prediction)
        self.memory.save("shap", shap)

        reasoning = self.reasoning.run(
            prediction,
            shap,
            signals,
        )
        self.memory.save("reasoning", reasoning)

        ehr = self.ehr.run(reasoning)
        self.memory.save("ehr", ehr)

        doctor = self.doctor.run(ehr)
        patient = self.patient.run(ehr)

        self.pdf.run(doctor, patient)

        self.logger.log("AI Agent Finished")

        return ehr