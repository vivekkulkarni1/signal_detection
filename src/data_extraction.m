function [data_north,data_east,data_vertical,t1,t2,t3,fs] = data_extraction(x)

data_north = [];
data_east = [];
data_vertical = [];
t1 = [];
t2 = [];
t3 = [];
fs = zeros(1,length(x));

for i = 1:length(x)
    if strcmp(x(i).ChannelIdentifier, 'BH1')
        
        data_north = [data_north; x(i).d];
        t1 = [t1; x(i).t]; % Times may not be same for all channels
        
    elseif strcmp(x(i).ChannelIdentifier, 'BHN')
        
        data_north = [data_north; x(i).d];
        t1 = [t1; x(i).t]; % Times may not be same for all channels
        
    elseif strcmp(x(i).ChannelIdentifier, 'HHN')
        
        data_north = [data_north; x(i).d];
        t1 = [t1; x(i).t]; % Times may not be same for all channels
        
    elseif strcmp(x(i).ChannelIdentifier, 'SHN')
        
        data_north = [data_north; x(i).d];
        t1 = [t1; x(i).t]; % Times may not be same for all channels
        
    elseif strcmp(x(i).ChannelIdentifier,'BH2')
        
        data_east = [data_east; x(i).d];
        t2 = [t2; x(i).t];
        
     elseif strcmp(x(i).ChannelIdentifier,'BHE')
        
        data_east = [data_east; x(i).d];
        t2 = [t2; x(i).t];
        
    elseif strcmp(x(i).ChannelIdentifier,'HHE')
        
        data_east = [data_east; x(i).d];
        t2 = [t2; x(i).t];
        
    elseif strcmp(x(i).ChannelIdentifier,'SHE')
        
        data_east = [data_east; x(i).d];
        t2 = [t2; x(i).t];
     
    elseif strcmp(x(i).ChannelIdentifier, 'BHZ')
        
        data_vertical = [data_vertical; x(i).d];
        t3 = [t3; x(i).t];
        
    elseif strcmp(x(i).ChannelIdentifier, 'HHZ')
        
        data_vertical = [data_vertical; x(i).d];
        t3 = [t3; x(i).t];
        
    elseif strcmp(x(i).ChannelIdentifier, 'SHZ')
        
        data_vertical = [data_vertical; x(i).d];
        t3 = [t3; x(i).t];     
    end
    
    fs(i) = x(i).SampleRate; % Sampling rate is same for all channels in this case; if they are not, then it must be accounted for
    
end

% Detrending the data.

data_north = detrend(data_north,0);
data_east = detrend(data_east,0);
data_vertical = detrend(data_vertical,0);

end


