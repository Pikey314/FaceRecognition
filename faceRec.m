clear all; 
close all; 
clc

faceRecognizer('50cent.jpg','50Cent')

function faceRecognizer(arg1,arg2)

figure(1);

Samik1 = imresize(double(rgb2gray(imread('Samik1','jpg'))),[60 40]);
Samik2 = imresize(double(rgb2gray(imread('Samik2','jpg'))),[60 40]);
Samik3 = imresize(double(rgb2gray(imread('Samik3','jpg'))),[60 40]);
Samik4 = imresize(double(rgb2gray(imread('Samik4','jpg'))),[60 40]);

Irek1 = imresize(double(rgb2gray(imread('I1','jpg'))),[60 40]);
Irek2 = imresize(double(rgb2gray(imread('I2','jpg'))),[60 40]);
Irek3 = imresize(double(rgb2gray(imread('I3','jpg'))),[60 40]);
Irek4 = imresize(double(rgb2gray(imread('I4','jpg'))),[60 40]);

Ania1 = imresize(double(rgb2gray(imread('A1','jpg'))),[60 40]);
Ania2 = imresize(double(rgb2gray(imread('A2','jpg'))),[60 40]);
Ania3 = imresize(double(rgb2gray(imread('A3','jpg'))),[60 40]);
Ania4 = imresize(double(rgb2gray(imread('A4','jpg'))),[60 40]);


Renia1 = imresize(double(rgb2gray(imread('R1','jpg'))),[60 40]);
Renia2 = imresize(double(rgb2gray(imread('R2','jpg'))),[60 40]);
Renia3 = imresize(double(rgb2gray(imread('R3','jpg'))),[60 40]);
Renia4 = imresize(double(rgb2gray(imread('R4','jpg'))),[60 40]);

Jedrzej1 = imresize(double(rgb2gray(imread('JJ1','jpg'))),[60 40]);
Jedrzej2 = imresize(double(rgb2gray(imread('JJ2','jpg'))),[60 40]);
Jedrzej3 = imresize(double(rgb2gray(imread('JJ3','jpg'))),[60 40]);
Jedrzej4 = imresize(double(rgb2gray(imread('JJ4','jpg'))),[60 40]);


Maciek1 = imresize(double(rgb2gray(imread('M1','jpg'))),[60 40]);
Maciek2 = imresize(double(rgb2gray(imread('M2','jpg'))),[60 40]);
Maciek3 = imresize(double(rgb2gray(imread('M3','jpg'))),[60 40]);
Maciek4 = imresize(double(rgb2gray(imread('M4','jpg'))),[60 40]);


Bodo1 = imresize(double((imread('B_1','bmp'))),[60 40]);
Bodo2 = imresize(double((imread('B_2','bmp'))),[60 40]);
Bodo3 = imresize(double((imread('B_3','bmp'))),[60 40]);
Bodo4 = imresize(double((imread('B_4','bmp'))),[60 40]);


Smos1 = imresize(double((imread('SM_1','bmp'))),[60 40]);
Smos2 = imresize(double((imread('SM_2','bmp'))),[60 40]);
Smos3 = imresize(double((imread('SM_3','bmp'))),[60 40]);
Smos4 = imresize(double((imread('SM_4','bmp'))),[60 40]);


Florek1 = imresize(double((imread('F_1','bmp'))),[60 40]);
Florek2 = imresize(double((imread('F_2','bmp'))),[60 40]);
Florek3 = imresize(double((imread('F_3','bmp'))),[60 40]);
Florek4 = imresize(double((imread('F_4','bmp'))),[60 40]);


Zelmer1 = imresize(double((imread('ZE_1','bmp'))),[60 40]);
Zelmer2 = imresize(double((imread('ZE_2','bmp'))),[60 40]);
Zelmer3 = imresize(double((imread('ZE_3','bmp'))),[60 40]);
Zelmer4 = imresize(double((imread('ZE_4','bmp'))),[60 40]);


subplot(4,10,1), pcolor(flipud(Samik1)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,2), pcolor(flipud(Samik2)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,3), pcolor(flipud(Samik3)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,4), pcolor(flipud(Samik4)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);

subplot(4,10,5), pcolor(flipud(Bodo1)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,6), pcolor(flipud(Bodo2)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,7), pcolor(flipud(Bodo3)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,8), pcolor(flipud(Bodo4)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);

subplot(4,10,9), pcolor(flipud(Smos1)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,10), pcolor(flipud(Smos2)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,11), pcolor(flipud(Smos3)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,12), pcolor(flipud(Smos4)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);

subplot(4,10,13), pcolor(flipud(Florek1)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,14), pcolor(flipud(Florek2)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,15), pcolor(flipud(Florek3)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,16), pcolor(flipud(Florek4)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);

subplot(4,10,17), pcolor(flipud(Zelmer1)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,18), pcolor(flipud(Zelmer2)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,19), pcolor(flipud(Zelmer3)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,20), pcolor(flipud(Zelmer4)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);

subplot(4,10,21), pcolor(flipud(Irek1)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,22), pcolor(flipud(Irek2)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,23), pcolor(flipud(Irek3)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,24), pcolor(flipud(Irek4)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);

subplot(4,10,25), pcolor(flipud(Maciek1)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,26), pcolor(flipud(Maciek2)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,27), pcolor(flipud(Maciek3)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,28), pcolor(flipud(Maciek4)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);

subplot(4,10,29), pcolor(flipud(Jedrzej1)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,30), pcolor(flipud(Jedrzej2)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,31), pcolor(flipud(Jedrzej3)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,32), pcolor(flipud(Jedrzej4)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);

subplot(4,10,33), pcolor(flipud(Ania1)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,34), pcolor(flipud(Ania2)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,35), pcolor(flipud(Ania3)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,36), pcolor(flipud(Ania4)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);

subplot(4,10,37), pcolor(flipud(Renia1)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,38), pcolor(flipud(Renia2)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,39), pcolor(flipud(Renia3)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(4,10,40), pcolor(flipud(Renia4)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);

figure(2);

AverageSamik = (Samik1+Samik2+Samik3+Samik4)/4;
AverageBodo = (Bodo1+Bodo2+Bodo3+Bodo4)/4;
AverageSmos = (Smos1+Smos2+Smos3+Smos4)/4;
AverageFlorek = (Florek1+Florek2+Florek3+Florek4)/4;
AverageZelmer = (Zelmer1+Zelmer2+Zelmer3+Zelmer4)/4;

AverageIrek = (Irek1+Irek2+Irek3+Irek4)/4;
AverageMaciek = (Maciek1+Maciek2+Maciek3+Maciek4)/4;
AverageJedrzej = (Jedrzej1+Jedrzej2+Jedrzej3+Jedrzej4)/4;
AverageAnia = (Ania1+Ania2+Ania3+Ania4)/4;
AverageRenia = (Renia1+Renia2+Renia3+Renia4)/4;


subplot(2,5,1), pcolor(flipud(AverageSamik)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(2,5,2), pcolor(flipud(AverageBodo)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(2,5,3), pcolor(flipud(AverageSmos)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(2,5,4), pcolor(flipud(AverageFlorek)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(2,5,5), pcolor(flipud(AverageZelmer)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);

subplot(2,5,6), pcolor(flipud(AverageIrek)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(2,5,7), pcolor(flipud(AverageMaciek)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(2,5,8), pcolor(flipud(AverageJedrzej)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(2,5,9), pcolor(flipud(AverageAnia)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(2,5,10), pcolor(flipud(AverageRenia)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);


D = [reshape(Samik1,1,60*40)
    reshape(Samik2,1,60*40)
    reshape(Samik3,1,60*40)
    reshape(Samik4,1,60*40)
    reshape(Bodo1,1,60*40)
    reshape(Bodo2,1,60*40)
    reshape(Bodo3,1,60*40)
    reshape(Bodo4,1,60*40)
    reshape(Smos1,1,60*40)
    reshape(Smos2,1,60*40)
    reshape(Smos3,1,60*40)
    reshape(Smos4,1,60*40)
    reshape(Florek1,1,60*40)
    reshape(Florek2,1,60*40)
    reshape(Florek3,1,60*40)
    reshape(Florek4,1,60*40)
    reshape(Zelmer1,1,60*40)
    reshape(Zelmer2,1,60*40)
    reshape(Zelmer3,1,60*40)
    reshape(Zelmer4,1,60*40)
    reshape(Irek1,1,60*40)
    reshape(Irek2,1,60*40)
    reshape(Irek3,1,60*40)
    reshape(Irek4,1,60*40)
    reshape(Maciek1,1,60*40)
    reshape(Maciek2,1,60*40)
    reshape(Maciek3,1,60*40)
    reshape(Maciek4,1,60*40)
    reshape(Jedrzej1,1,60*40)
    reshape(Jedrzej2,1,60*40)
    reshape(Jedrzej3,1,60*40)
    reshape(Jedrzej4,1,60*40)
    reshape(Ania1,1,60*40)
    reshape(Ania2,1,60*40)
    reshape(Ania3,1,60*40)
    reshape(Ania4,1,60*40)
    reshape(Renia1,1,60*40)
    reshape(Renia2,1,60*40)
    reshape(Renia3,1,60*40)
    reshape(Renia4,1,60*40)];

A = (D')*(D);
size(A)

[V,D] = eigs(A,20,'lm');

figure(3);
subplot(3,4,1), face1=reshape(V(:,1),60,40); pcolor(flipud(face1)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(3,4,2), face1=reshape(V(:,2),60,40); pcolor(flipud(face1)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(3,4,3), face1=reshape(V(:,3),60,40); pcolor(flipud(face1)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(3,4,4), face1=reshape(V(:,4),60,40); pcolor(flipud(face1)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(3,4,5), face1=reshape(V(:,5),60,40); pcolor(flipud(face1)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);

subplot(3,4,6), face1=reshape(V(:,6),60,40); pcolor(flipud(face1)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(3,4,7), face1=reshape(V(:,7),60,40); pcolor(flipud(face1)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(3,4,8), face1=reshape(V(:,8),60,40); pcolor(flipud(face1)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(3,4,9), face1=reshape(V(:,9),60,40); pcolor(flipud(face1)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(3,4,10), face1=reshape(V(:,10),60,40); pcolor(flipud(face1)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);


subplot(3,4,11), semilogy(diag(D), 'ko', 'Linewidth',[2])
set(gca,'Fontsize',[14])

figure(4)
vecSamik=reshape(AverageSamik,1,60*40);
vecBodo=reshape(AverageBodo,1,60*40);
vecSmos=reshape(AverageSmos,1,60*40);
vecFlorek=reshape(AverageFlorek,1,60*40);
vecZelmer=reshape(AverageZelmer,1,60*40);

vecIrek=reshape(AverageIrek,1,60*40);
vecMaciek=reshape(AverageMaciek,1,60*40);
vecJedrzej=reshape(AverageJedrzej,1,60*40);
vecAnia=reshape(AverageAnia,1,60*40);
vecRenia=reshape(AverageRenia,1,60*40);

projSamik = vecSamik*V;
projBodo=vecBodo*V;
projSmos=vecSmos*V;
projFlorek=vecFlorek*V;
projZelmer=vecZelmer*V;

projIrek=vecIrek*V;
projMaciek=vecMaciek*V;
projJedrzej=vecJedrzej*V;
projAnia=vecAnia*V;
projRenia=vecRenia*V;

subplot(2,5,1), bar(projSamik(2:20)), set(gca, 'Xlim',[0,20],'Ylim',[-2000,2000],'Xtick',[],'Ytick',[])
 text(12,-1700, 'Samik','Fontsize',[15])
 subplot(2,5,2), bar(projBodo(2:20)), set(gca, 'Xlim',[0,20],'Ylim',[-2000,2000],'Xtick',[],'Ytick',[])
 text(12,-1700, 'Bodo','Fontsize',[15])
 subplot(2,5,3), bar(projSmos(2:20)), set(gca, 'Xlim',[0,20],'Ylim',[-2000,2000],'Xtick',[],'Ytick',[])
 text(12,-1700, 'Smos','Fontsize',[15])
 subplot(2,5,4), bar(projFlorek(2:20)), set(gca, 'Xlim',[0,20],'Ylim',[-2000,2000],'Xtick',[],'Ytick',[])
 text(12,-1700, 'Florek','Fontsize',[15])
 subplot(2,5,5), bar(projZelmer(2:20)), set(gca, 'Xlim',[0,20],'Ylim',[-2000,2000],'Xtick',[],'Ytick',[])
 text(12,-1700, 'Zelmer','Fontsize',[15])
 
 subplot(2,5,6), bar(projIrek(2:20)), set(gca, 'Xlim',[0,20],'Ylim',[-2000,2000],'Xtick',[],'Ytick',[])
 text(12,-1700, 'Irek','Fontsize',[15])
 subplot(2,5,7), bar(projMaciek(2:20)), set(gca, 'Xlim',[0,20],'Ylim',[-2000,2000],'Xtick',[],'Ytick',[])
 text(12,-1700, 'Maciek','Fontsize',[15])
 subplot(2,5,8), bar(projJedrzej(2:20)), set(gca, 'Xlim',[0,20],'Ylim',[-2000,2000],'Xtick',[],'Ytick',[])
 text(12,-1700, 'Jedrzej','Fontsize',[15])
 subplot(2,5,9), bar(projAnia(2:20)), set(gca, 'Xlim',[0,20],'Ylim',[-2000,2000],'Xtick',[],'Ytick',[])
 text(12,-1700, 'Ania','Fontsize',[15])
 subplot(2,5,10), bar(projRenia(2:20)), set(gca, 'Xlim',[0,20],'Ylim',[-2000,2000],'Xtick',[],'Ytick',[])
 text(12,-1700, 'Renia','Fontsize',[15])
 
 
 %testing
 
 extension = arg1(end-2:end);
 if (extension == 'jpg')
     T1 = imresize( double( rgb2gray( imread(arg1(1:end-4),'jpg'))),[60 40]);
 elseif (extension == 'bmp')
     T1 = imresize( double(( imread(arg1(1:end-4),'bmp'))),[60 40]);
 else
      disp('Wrong file extension: should be jpg or bmp')
      return
 end
     
 
 vec1 = reshape(T1,1,60*40);
 
 projTest=vec1*V;
 
 recon1=V*projTest' ; rec1=reshape(recon1,60,40);
 
 figure(5)
subplot(1,2,1), pcolor(flipud(rec1)), shading interp, colormap(gray), set(gca,'Xtick',[],'Ytick',[]);
subplot(1,2,2), bar(projTest(2:20)), set(gca, 'Xlim',[0,20],'Ylim',[-2000,2000],'Xtick',[],'Ytick',[])
 text(12,-1700, arg2,'Fontsize',[15])

 
 winnerSamik = sum(abs(projSamik-projTest));
 winnerBodo = sum(abs(projBodo-projTest));
 winnerSmos = sum(abs(projSmos-projTest));
 winnerFlorek = sum(abs(projFlorek-projTest));
 winnerZelmer = sum(abs(projZelmer-projTest));
 
 winnerIrek = sum(abs(projIrek-projTest));
 winnerMaciek = sum(abs(projMaciek-projTest));
 winnerJedrzej = sum(abs(projJedrzej-projTest));
 winnerAnia = sum(abs(projAnia-projTest));
 winnerRenia = sum(abs(projRenia-projTest));
 
 winnersVector = [winnerSamik winnerBodo winnerSmos winnerFlorek winnerZelmer winnerIrek winnerMaciek winnerJedrzej winnerAnia winnerRenia];
 figure(6);
 subplot(1,1,1), bar(winnersVector), set(gca,'xticklabel',{'Samik','Bobo','Smos','Florek','Zelmer','Irek','Maciek','Jedrzej','Ania','Renia'});
winnerOfTest = min(winnersVector);

if (winnerOfTest == winnerSamik)
    disp('Winner is SAMIK');
elseif (winnerOfTest == winnerBodo)
    disp('Winner is Bodo');
elseif (winnerOfTest == winnerSmos)
    disp('Winner is Smos');
elseif (winnerOfTest == winnerFlorek)
    disp('Winner is Florek');
elseif (winnerOfTest == winnerIrek)
    disp('Winner is Irek');
elseif (winnerOfTest == winnerMaciek)
    disp('Winner is Maciek');
elseif (winnerOfTest == winnerJedrzej)
    disp('Winner is Jedrzej');
elseif (winnerOfTest == winnerAnia)
    disp('Winner is Ania');
elseif (winnerOfTest == winnerRenia)
    disp('Winner is Renia');
else
    disp('Winner is Zelmer');
   
end

    
    
end

    
 
 
