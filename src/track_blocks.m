%
% track_blocks.m
%   track blocks over time using a Kalman filter
%
% AUTHOR
%   Jonathan D. Jones
%

close all; clear; clc

dir_path = '/Users/jonathan/block_coords';
dir_contents = dir(fullfile(dir_path, '*.csv'));
bytes = [dir_contents.bytes];
filenames = {dir_contents.name};
% FIXME: skip these files in the tracking loop instead?
filenames = filenames(bytes > 0);
num_files = size(filenames, 2);

tracks = struct(...
    'id', {}, ...
    'kalmanFilter', {}, ...
    'age', {}, ...
    'totalVisibleCount', {}, ...
    'consecutiveInvisibleCount', {}, ...
    'colorHist', {});

% Detect phase
init_frame_fn = fullfile(dir_path, filenames{1});
M = csvread(init_frame_fn);
num_tracks = size(M, 1);
% Contents of M:
%   0 -- block center x
%   1 -- block center y
%   2 -- orientation angle
%   3 -- block length
%   4 -- block width
%   5 -- block depth (ignore)
%   6 -- red count
%   7 -- blue count
%   8 -- green count
%   9 -- yellow count
%   10 - grey count
nextID = 0;
obj_centroids = cell(num_tracks, 1);
obj_colorhists = cell(num_tracks, 1);
for i = 1:num_tracks
    nextID = nextID + 1;
    % Set up a kalman filter for each detected object
    % Create a Kalman filter object.
    centroid = M(i,1:2);
    color_hist = M(i,end-4:end);
    kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
        centroid, [200, 50], [100, 25], 100);
    newTrack = struct(...
        'id', nextID, ...
        'kalmanFilter', kalmanFilter, ...
        'age', 1, ...
        'totalVisibleCount', 1, ...
        'consecutiveInvisibleCount', 0, ...
        'colorHist', color_hist);
    tracks(end + 1) = newTrack;
    
    obj_centroids{i} = zeros(num_files, 2);
    obj_centroids{i}(1,:) = centroid;
    
    obj_colorhists{i} = zeros(num_files, 5);
    obj_colorhists{i}(1,:) = color_hist;
end

%%
% Track phase
tol = 10;
for i = 2:num_files
    % Load file
    frame_fn = fullfile(dir_path, filenames{i});
    M = csvread(frame_fn);
    num_obs = size(M, 1);
    centroids = M(:,1:2);
    color_hists = M(:,end-4:end);
    % Calculate distance to each predicted centroid
    cost = zeros(num_tracks, num_obs);
    for j = 1:num_tracks
        predicted_centroid = predict(tracks(j).kalmanFilter);
        cost(j,:) = distance(tracks(j).kalmanFilter, centroids);
    end
    % Assign each centroid to nearest predicted centroid and update
    [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, tol);
    for j = 1:size(assignments, 1)
        k_idx = assignments(j,1);
        o_idx = assignments(j,2);
        correct(tracks(k_idx).kalmanFilter, centroids(o_idx,:));
        obj_centroids{k_idx}(i,:) = centroids(o_idx,:);
        obj_colorhists{k_idx}(i,:) = color_hists(o_idx,:);
        tracks(k_idx).colorHist = tracks(k_idx).colorHist ...
                                + color_hists(o_idx,:);
    end
    
end

colors = {'r', 'g', 'b', 'y', 'k'};
%%
% Display results
figure();
set(gca,'Ydir','reverse')
hold on
output_dir = '/Users/jonathan/filtered_coords';
for i = 1:length(obj_centroids)
    
    color_hist = tracks(i).colorHist;
    [~, idxs] = sort(color_hist, 'descend');
    color = colors{idxs(1)};
    %if idxs(1) == 5
    %    color = colors{idxs(2)};
    %end
    
    c = obj_centroids{i};
    c = c(c(:,1) > 0 & c(:,2) > 0,:);
    plot(c(:,2), c(:,1), color)
    scatter(c(:,2), c(:,1), color)
    
    data = [obj_centroids{i}, obj_colorhists{i}];
    fn = fullfile(output_dir, [num2str(i), '.csv']);
    csvwrite(fn, data)
end
