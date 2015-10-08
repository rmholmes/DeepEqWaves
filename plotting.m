base = 'snapshots/';
base2 = 'snapshots_s1';
np = 20;
fnames = cell(np,1);
for ii=1:np
fnames{ii} = [base base2 '/' base2 '_p' num2str(ii-1) '.h5'];
end

figure;
set(gcf,'Position',[137    54   855   607]);
%set(gcf,'Position',get(0,'ScreenSize'));
%for ti = 1:28
for ii=1:length(fnames)
    subplot(length(fnames),1,length(fnames)-ii+1);
    var = h5read(fnames{ii},'/tasks/v');
    [X,Y] = ndgrid(1:length(var(:,1,1)),...
                   1:length(var(1,:,1)));
    %    pcolPlot(Y,X,var(:,:,ti));
    pcolPlot(Y,X,var(:,:,end));
    caxis([-1 1]*1e-2);
    %caxis([-1 1]*1e-4);
    if (ii>1)
        posn = [pos(1) pos(2)+pos(4) pos(3) pos(4)];
        set(gca,'Position',posn);
        set(gca,'xticklabel',[]);
    end
    pos = get(gca,'Position');
    colorbar off;
    
end
%export_fig(['frame' sprintf('%02d',ti) '.png'],'-painters');
%clf;
%end
