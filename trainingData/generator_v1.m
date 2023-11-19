clear all
close all
"--------------------------------------------------"
for num = 1:1000
num
numberOfSegments = 30;
UEAntPosDif = 0.1;
UESpeed = 4;
prevDeg = randi(361) - 1;
track = qd_track('linear');%, rand*0.2+0.05, -prevDeg/180 * pi); %First segment
track.initial_position = [rand*1000; rand*1000; 2]; 
len = 10;
for segN = 1:numberOfSegments
    len = rand*0.2+0.05;
    newDeg = randi(361) - 1;
    %3D line
    c = len*exp(-1j*newDeg/180*pi);
    track.positions = [track.positions,...
    [ track.positions(1,end) + real(c); track.positions(2,end) + imag(c); 2 + (rand-0.5)*0.2 ]];%track.positions(3,end) + (rand-0.5)*0.2 ]];
    prevDeg = newDeg;
end
track.positions = track.positions(:,3:end);


% set(0,'defaultTextFontSize', 18)                      	% Default Font Size
% set(0,'defaultAxesFontSize', 18)                     	% Default Font Size
% set(0,'defaultAxesFontName','Times')               	    % Default Font Type
% set(0,'defaultTextFontName','Times')                 	% Default Font Type
% set(0,'defaultFigurePaperPositionMode','auto')       	% Default Plot position
% set(0,'DefaultFigurePaperType','<custom>')             	% Default Paper Type
% set(0,'DefaultFigurePaperSize',[14.5 7.8])            	% Default Paper Size
%track.calc_orientation;                                     % Calculate the receiver orientation

l = qd_layout;                                          % New layout
%l.update_rate = 0.001;
[~,l.rx_track] = interpolate( track.copy,'distance',0.01 );  % Interpolate and assign track to layout

l.rx_track.positions(1, :) = lowpass(l.rx_track.positions(1, :), 0.01, 1000);
l.rx_track.positions(2, :) = lowpass(l.rx_track.positions(2, :), 0.01, 1000);
l.rx_track.positions(3, :) = lowpass(l.rx_track.positions(3, :), 0.01, 1000);
l.rx_track.positions = l.rx_track.positions(:, 10:end-10);
l.rx_track.calc_orientation;

%plot3(l.rx_track.positions(1, :), l.rx_track.positions(2, :), l.rx_track.positions(3, :))%l.visualize([],[],0);                                   % Plot
%axis equal

%zaszumianie
BSantenna = qd_arrayant('3gpp-3d', 2, 2, 3.6e9, 6, 8);
l.rx_array = qd_arrayant('dipole');
l.rx_array.copy_element(1,2);
l.rx_array.rotate_pattern(90, 'y', 2);
l.rx_array.copy_element(1,3);
l.rx_array.copy_element(2,4);
l.rx_array.element_position(:,2) = [UEAntPosDif; 0; 0];
l.rx_array.element_position(:,3) = [0; UEAntPosDif; 0];
l.rx_array.element_position(:,4) = [UEAntPosDif; UEAntPosDif; 0];
%%l.rx_array.Fa(:,:,2) = 0;
%l.rx_array.Fb(:,:,2) = 1;
l.rx_array.center_frequency = 3.6e9;
l.simpar.center_frequency = 3.6e9;
l.tx_position = [0;500;rand*10+3];
l.tx_track.orientation = [0; 0; 0];
l.tx_array = BSantenna; %qd_arrayant('dipole');
l.tx_array.center_frequency = 3.6e9;
%l.tx_array.visualize

l.rx_track.scenario = 'QuaDRiGa_NTN_Urban_NLOS';
l.rx_track.movement_profile = [0, get_length(l.rx_track)/UESpeed; 0, get_length(l.rx_track)];
timePerSnapshot = get_length(l.rx_track)/UESpeed / l.rx_track.no_snapshots;
% dist = l.rx_track.interpolate('time', 0.1);
% time  = ( 0:numel(dist) - 2 )*0.1;
% speed = diff( dist ) * 10;

[c, builder] = l.get_channels;
%builder.visualize_clusters();

% % Calculate the beam footprint
% set(0,'DefaultFigurePaperSize',[14.5 7.8])              % Adjust paper size for plot
% [map,x_coords,y_coords]=l.power_map('QuaDRiGa_NTN_Urban_NLOS','detailed',0.01,-1,1,-1,1);
% P = 10*log10( map{:}(:,:,1) ) + 50;                     % RX copolar power @ 50 dBm TX power
% l.visualize([],[],0);                                   % Plot layout
% axis([-1,1,-1,1]);                              % Axis
% hold on
% imagesc( x_coords, y_coords, P );                       % Plot the received power
% hold off
% 
% colorbar('South')                                       % Show a colorbar
% colmap = colormap;
% colormap( colmap*0.5 + 0.5 );                           % Adjust colors to be "lighter"
% axis equal
% set(gca,'XTick',(-5:5)*1e6);
% set(gca,'YTick',(-5:5)*1e6);
% caxis([-150,-90])
% set(gca,'layer','top')                                  % Show grid on top of the map
% title('Beam footprint in dBm');                         % Set plot title



% pow  = 10*log10(  reshape(sum(abs(c.coeff(:,:,:,:)).^2,3), 1, [])  );    % Calculate the power
% time = (0:c.no_snap-1)*0.01;                            % Vector with time samples
% figure(3)
% hold on; plot(time,pow'+50); hold off;

freq_response = fr(c, 100e6, 3276);

gyroX = l.rx_track.orientation(3, 2:end)-l.rx_track.orientation(3, 1:end-1); %orientation from above
gyroZ = l.rx_track.orientation(2, 2:end)-l.rx_track.orientation(2, 1:end-1); %up or down
gyro = [gyroX; gyroZ];
save(['freq_resp', num2str(num)], 'freq_response');
save(['gyro', num2str(num)], 'gyro');
clear all
close all
end