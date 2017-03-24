%  This is a snippet for testing the heatmap from the shifted outputs
      alpha=0.3;
      hm = zeros(15,15);
      count_hm = zeros(15,15);
      f_0_0 = rand(5, 5); f_0_0 = exp(alpha*f_0_0)/sum(exp(alpha*f_0_0(:)));
      f_1_0 = rand(5, 5);  f_1_0 = exp(alpha*f_1_0)/sum(exp(alpha*f_1_0(:)));
      f_m1_0 = rand(5 ,5); f_m1_0 = exp(alpha*f_m1_0)/sum(exp(alpha*f_m1_0(:)));
      f_0_m1 = rand(5,5);  f_0_m1 = exp(alpha*f_0_m1)/sum(exp(alpha*f_0_m1(:)));
      f_0_1 = rand(5,5); f_0_1 = exp(alpha*f_0_1)/sum(exp(alpha*f_0_1(:)));
      
      f_cell = {f_0_0, f_1_0, f_m1_0, f_0_m1, f_0_1};
      %This v_x and v_y are arrays containing no. describing how much shift they have in x and y directions respectively.
      v_x = [0 1 -1 0 0];
      v_y = [0 0 0 -1 1];
      for k=1:5
        delta_x = v_x(k);
        delta_y = v_y(k);
        f = f_cell{k};
        for x=1:5
            for y=1:5
                i_x = 1+3*(x-1) - delta_x; i_x = max(i_x,1);
                if(x==1)
                    i_x=1;
                end
                i_y = 1+3*(y-1) - delta_y; i_y = max(i_y,1);
                if(y==1)
                    i_y =1;
                end
                f_x = 3*x-delta_x; f_x = min(15,f_x);
                if(x==5)
                    f_x = 15;
                end
                f_y = 3*y-delta_y; f_y = min(15,f_y);
                if(y==5)
                    f_y = 15;
                end
                hm(i_x:f_x,i_y:f_y) = hm(i_x:f_x,i_y:f_y)+f(x,y);
                count_hm(i_x:f_x,i_y:f_y) = count_hm(i_x:f_x,i_y:f_y)+1;
            end
        end
      end
      
      hm_base = hm./count_hm; %count_hm is for averaging out
      sum(hm_base(:))