from utils.evaluation_metrics.crps import evaluate_crps_from_h5,crps_ensemble
from utils.evaluation_metrics.csi import evaluate_csi_from_h5,csi
try:
    # evaluate_crps_from_h5("/home1/ppatel2025/nowcasting2/nowcaster_results/forecast_results/steps_ensemble_20/test/results.h5", "/home1/ppatel2025/nowcasting2/nowcaster_results/model_evaluation_results",10)
    evaluate_csi_from_h5("nowcaster_results/forecast_results/steps_ensemble_20/test/results.h5",batch_size=10)


except:
    print("failed")

