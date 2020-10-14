import numpy as np
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def get_complete_ts_predictions(time_series, model, input_window_length, forecast_length):
    
    ground_truth_values = []
    predicted_values = []
    
    first_possible_input_index = 0
    last_possible_input_index = (len(time_series) 
                                - input_window_length 
                                - forecast_length)
    
    gap_between_forecasts = forecast_length
    
    for start_index in tqdm(range(first_possible_input_index,
                       last_possible_input_index,
                       gap_between_forecasts)):
        
        model_input = time_series[start_index:start_index + input_window_length]
        model_prediction = model.predict(model_input)
        
        ground_truth =  time_series[start_index + input_window_length
                                    : start_index + input_window_length + forecast_length]
        
        ground_truth_values.extend(ground_truth)
        if(type(model_prediction) == np.ndarray):
            predicted_values.extend(model_prediction.tolist())
        else:
            predicted_values.extend(model_prediction)

        
    return ground_truth_values, predicted_values

def get_complete_ts_anomaly_score(ground_truth_values, 
                      predicted_values, 
                      epsilon, 
                      quantile_scaling,
                      threshold):
    
    ground_truth_array = np.squeeze(np.array(ground_truth_values)) + epsilon
    predicted_array = np.squeeze(np.array(predicted_values)) + epsilon
    
    difference = ground_truth_array - predicted_array
    absolute_difference = np.abs(difference)
    
    percent_error = difference / ground_truth_array
    forecast_percent_error = difference / predicted_array
    
    anomaly_score = percent_error
    
    if(quantile_scaling):
        quantile_scaler = QuantileTransformer(output_distribution='normal')
        anomaly_score = quantile_scaler.fit_transform(anomaly_score)
    
    if(threshold is not None):
        anomaly_score[anomaly_score < threshold] = 0
    
    return anomaly_score