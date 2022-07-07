function Z = points_to_depth(P, rows, cols, kernal_size, smoothNO)

    %normalize points cneter at 0
    P_norm(:,1) = ( P(:,1)-min(min(P(:,1:2))) )/( max(max(P(:,1:2)))-min(min(P(:,1:2))) );
    shift1 =  mean(P_norm(:,1));
    P_norm(:,1) = P_norm(:,1) - shift1;
    P_norm(:,2) = ( P(:,2)-min(min(P(:,1:2))) )/( max(max(P(:,1:2)))-min(min(P(:,1:2))) );
    shift2 = mean(P_norm(:,2));
    P_norm(:,2) = P_norm(:,2) - shift2;
    P_norm(:,3) = (P(:,3)-min(P(:,3)))/(max(P(:,3))-min(P(:,3)))-0.5;
    %scatter3(P_norm(:,1),P_norm(:,2),P_norm(:,3))
    
    %normalize points to image dimension
    PX = ceil(( P_norm(:,1)-min(min(P_norm(:,1:2))) )/( max(max(P_norm(:,1:2)))-min(min(P_norm(:,1:2))) )*(cols-1))+1;
    PY = ceil(( P_norm(:,2)-min(min(P_norm(:,1:2))) )/( max(max(P_norm(:,1:2)))-min(min(P_norm(:,1:2))) )*(rows-1))+1;
    %scatter3(PX,PY,P_norm(:,3))
    
    %fitting points to depth map
    Z = zeros(rows, cols);
    for i = 1:length(P_norm)
        if PY(i)>=1 && PY(i)<=rows && PX(i)>=1 && PX(i)<=cols && P_norm(i,3)>-0.5
            if Z(rows+1-PY(i), PX(i)) == 0 || P_norm(i,3) > Z(rows+1-PY(i), PX(i))
                Z(rows+1-PY(i), PX(i)) = P_norm(i,3);
            end
        end
    end
    %imshow(Z,[])
    
    %inconsistence check/hidden surface removal
    Z_clean = Z;
    for i = 1+kernal_size:rows-kernal_size
        for j = 1+kernal_size:cols-kernal_size
            if Z(i,j) ~= 0
                v = [];
                for vi = -kernal_size:kernal_size
                    for vj = -kernal_size:kernal_size
                        if ~(vi==0&&vj==0)
                            v = [v, Z(i+vi,j+vj)];
                        end
                    end
                end
                v = nonzeros(v);
                v = max(v);
                if ~isempty(v) && v-Z(i,j)>0.15
                    Z_clean(i,j) = 0;
                end
            end
        end
    end
    Z = Z_clean;
    %imshow(Z_clean,[])

    %dense image smoothing
    while(smoothNO>0)
        Z_smooth = Z;
        for i = 1+kernal_size:rows-kernal_size
            for j = 1+kernal_size:cols-kernal_size
                if Z(i,j) == 0
                    v = [];
                    for vi = -kernal_size:kernal_size
                        for vj = -kernal_size:kernal_size
                            if ~(vi==0&&vj==0)
                                v = [v, Z(i+vi,j+vj)];
                            end
                        end
                    end
                    v = nonzeros(v);
                    if isempty(v)
                        Z_smooth(i,j) = 0;
                    else
                        Z_smooth(i,j) = mean(v);
                    end
                end
            end
        end
        smoothNO = smoothNO-1;
        Z = Z_smooth;
    end
    %imshow(Z,[])
    
    
    %denorm Z to range 0-1
    Z = Z+0.5;
    %%Background
    Z(Z==0.5) = 0;
    
end