%% Data In-Out Path
%datapath = '../BU3DFE/';                %***requires editing
datapath = '/media/jshi/Kevin 6TB/BU3DFE/'
outpath_2d = '../Data/2D/';
outpath_3d = '../Data/Depth/';

%% Variables
No_female = 56;
No_male = 44;

%% Script for Ethnicity Extraction
ethnicityF = char(ones(No_female,2));
ethnicityM = char(ones(No_male,2));
for subjectID = 1:No_female
    data = dir([datapath 'F' num2str(subjectID,'%.4d') '/']);
    str = data(3).name;
    ethnicityF(subjectID,:) = str(11:12);
end
for subjectID = 1:No_male
    data = dir([datapath 'M' num2str(subjectID,'%.4d') '/']);
    str = data(3).name;
    ethnicityM(subjectID,:) = str(11:12);
end


%% Labelling expression and Ethnicity
expression = char(ones(6,2));
expression(1,:) = 'AN';
expression(2,:) = 'DI';
expression(3,:) = 'FE';
expression(4,:) = 'HA';
expression(5,:) = 'SA';
expression(6,:) = 'SU';

ethnicity = char(ones(6,2));
ethnicity(1,:) = 'WH';
ethnicity(2,:) = 'BL';
ethnicity(3,:) = 'LA';
ethnicity(4,:) = 'AE';
ethnicity(5,:) = 'AM';
ethnicity(6,:) = 'IN';

%% Female
for subjectID = 1:No_female
    ethnicityID = find(ethnicity==ethnicityF(subjectID,:));
    ethnicityID = ethnicityID(1);
    for expressionID = 1:6
        for intensityID = 1:4
            disp([subjectID, expressionID, intensityID])
            %2D
            filename = ['F' num2str(subjectID,'%.4d') '/' 'F' num2str(subjectID,'%.4d') '_' expression(expressionID,:) num2str(intensityID,'%.2d') ethnicityF(subjectID,:) '_F2D.bmp'];
            
            if(~isfolder([outpath_2d filename(1:6)]))
                mkdir([outpath_2d filename(1:6)])
            end
            copyfile([datapath filename],[outpath_2d filename])
            
            %imwrite(imresize(imread([datapath filename]),[224,224]),[outpath_2d filename])
            
            %3D
            %{
            filename = ['F' num2str(subjectID,'%.4d') '/' 'F' num2str(subjectID,'%.4d') '_' expression(expressionID,:) num2str(intensityID,'%.2d') ethnicityF(subjectID,:) '_F3D.wrl'];
            P = read_points_from_file([datapath filename]);
            
            Z = points_to_depth(P, 160, 160, 2, 1);
            if(~isfolder([outpath_3d filename(1:6)]))
                mkdir([outpath_3d filename(1:6)])
            end
            imwrite(Z,[outpath_3d filename(1:end-4) '.bmp'])
            %}
        end
    end
end

%% Male
for subjectID = 1:No_male
    ethnicityID = find(ethnicity==ethnicityM(subjectID,:));
    ethnicityID = ethnicityID(1);
    for expressionID = 1:6
        for intensityID = 1:4
            disp([subjectID, expressionID, intensityID])
            %2D
            filename = ['M' num2str(subjectID,'%.4d') '/' 'M' num2str(subjectID,'%.4d') '_' expression(expressionID,:) num2str(intensityID,'%.2d') ethnicityM(subjectID,:) '_F2D.bmp'];
            if(~isfolder([outpath_2d filename(1:6)]))
                mkdir([outpath_2d filename(1:6)])
            end
            copyfile([datapath filename],[outpath_2d filename])
            
            %3D
            %{
            filename = ['M' num2str(subjectID,'%.4d') '/' 'M' num2str(subjectID,'%.4d') '_' expression(expressionID,:) num2str(intensityID,'%.2d') ethnicityM(subjectID,:) '_F3D.wrl'];
            P = read_points_from_file([datapath filename]);

            Z = points_to_depth(P, 160, 160, 2, 1);
            if(~isfolder([outpath_3d filename(1:6)]))
                mkdir([outpath_3d filename(1:6)])
            end
            imwrite(Z,[outpath_3d filename(1:end-4) '.bmp'])
            %}
        end
    end
end
