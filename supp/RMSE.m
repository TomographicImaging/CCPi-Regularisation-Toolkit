function err = RMSE(signal1, signal2)
%RMSE Root Mean Squared Error

err = sum((signal1 - signal2).^2)/length(signal1);  % MSE
err = sqrt(err);                                    % RMSE

end