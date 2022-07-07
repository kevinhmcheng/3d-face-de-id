function P = read_points_from_file(filename)

    f = fopen(filename,'r');
    start = 0;
    finish = 0;
    i = 1;
    while ~finish % While the file is not ended
        str = fgets(f);

        if contains(str,'point [')
            start = 1;
            str = fgets(f);
            %fprintf('Start.\n');
        elseif start && contains(str,']')
            %fprintf('Completed.\n');
            finish = 1;
        end
        if start && ~finish
            A = split(str);
            P(i,:) = [str2num(A{2}), str2num(A{3}), str2num(A{4})];
            i = i + 1;
        end
    end
    fclose(f);

end