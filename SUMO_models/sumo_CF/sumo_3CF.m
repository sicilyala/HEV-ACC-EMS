clear;
x1314=1:1314;
x4619=1:4619;

data_k=importdata("mix2\Krauss_v1.mat");
data_i=importdata("mix2\IDM_v1.mat");
data_a=importdata("mix2\ACC_v1.mat");
data8=importdata("E:\SEU2\Program1\MADDPG-program\model10\mix2_v12_g\episode_data\data_ep499.mat"); %schedule#8
pddpg='E:\SEU2\FanYi\Ecological_driving_layer_copy\run_result_0.3_v1\ep320_dev.mat';  % ep324      
data_ddpg=importdata(pddpg);

dis_k=data_k.distance;
dis_i=data_i.distance;
dis_a=data_a.distance;
dis_8=data8.distance;
dis_ddpg=data_ddpg.dev_vector;
dis_ddpg=[14,dis_ddpg];

spd_k=data_k.spd;
spd_i=data_i.spd;
spd_a=data_a.spd;
spd_8=data8.spd;

acc_k=data_k.acc;
acc_i=data_i.acc;
acc_a=data_a.acc;
acc_8=data8.acc;
acc_ddpg=data_ddpg.acc;
acc_ddpg=[0,acc_ddpg];

mode=1;
switch mode
    case 1
        plot(x4619,dis_k,x4619,dis_i,'black',x4619,dis_a,x4619,dis_ddpg, ...
            x4619,dis_8,'red')
        ylim([0,180])
        xlim([0,4700])
        legend("Krauss","IDM","ACC-SUMO","ACC-DDPG","ACC-MADDPG",FontSize=11,FontName="Times New Rome")
        xlabel("time step / s",FontSize=14,FontName="Times New Rome")
        ylabel("Following Distance (m)",FontSize=14,FontName="Times New Rome")
    case 2
        plot(x1314,spd_i,x1314,spd_a,x1314,spd_50)
        ylim([0,20])
    case 3
        plot(x1314,acc_i,x1314,acc_a,x1314,acc_50)
        ylim([-2,1])
    case 4
        varacc_k=var(acc_k);
        varacc_i=var(acc_i);
        varacc_a=var(acc_a);
        varacc_8=var(abs(acc_8));
        varacc_ddpg=var(acc_ddpg);
        y=[varacc_k,varacc_i,varacc_a,varacc_ddpg,varacc_8];
        bar(y);
        ylim([0,0.2])
        set(gca,'XTickLabel',["Krauss","IDM","ACC-SUMO","ACC-DDPG","ACC-MADDPG"], ...
            FontSize=12,FontName="Times New Rome"); 
        xlabel("Different car-following models",FontSize=14,FontName="Times New Rome")
        ylabel("Variance of Acceleration",FontSize=14,FontName="Times New Rome")
        mycolor='#FF0000';
        text(1,varacc_k+0.01,num2str(varacc_k),color=mycolor,FontSize=12, ...
            FontName="Times New Rome",HorizontalAlignment="center")
        text(2,varacc_i+0.01,num2str(varacc_i),color=mycolor,FontSize=12, ...
            FontName="Times New Rome",HorizontalAlignment="center")
        text(3,varacc_a+0.01,num2str(varacc_a),color=mycolor,FontSize=12, ...
            FontName="Times New Rome",HorizontalAlignment="center")
        text(4,varacc_ddpg+0.01,num2str(varacc_ddpg),color=mycolor,FontSize=12, ...
            FontName="Times New Rome",HorizontalAlignment="center")
        text(5,varacc_8+0.01,num2str(0.13641),color=mycolor,FontSize=12, ...
            FontName="Times New Rome",HorizontalAlignment="center")
    case 5
        X = [acc_k;acc_i;acc_a;acc_50]';    
        % 画箱型图的数据放在一个变量中，每列数据出一个箱型图
        boxplot(X);
        set(gca,'XTickLabel',["Krauss","IDM","ACC-SUMO","ACC-DDPG"],FontSize=12,FontName="Times New Rome"); 
        %设置x轴坐标标注，对应7个刻度
        xlabel("Different car-following models",FontSize=14,FontName="Times New Rome")
        ylabel("Acceleration (m/s^{2})",FontSize=14,FontName="Times New Rome")
        grid on;

end







