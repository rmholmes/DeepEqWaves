base = 'snapshots/snapshots_s3/';
fnames = {[base 'snapshots_s3_p0.h5'],...
          [base 'snapshots_s3_p1.h5'],...
          [base 'snapshots_s3_p2.h5'],...
          [base 'snapshots_s3_p3.h5']};

figure;
set(gcf,'Position',get(0,'ScreenSize'));
for ti = 1:28
for ii=1:length(fnames)
    subplot(length(fnames),1,length(fnames)-ii+1);
    var = h5read(fnames{ii},'/tasks/v');
    [X,Y] = ndgrid(1:length(var(:,1,1)),...
                   1:length(var(1,:,1)));
    pcolPlot(Y,X,var(:,:,ti));
    caxis([-5 5]*1e-2);
    if (ii>1)
        posn = [pos(1) pos(2)+pos(4) pos(3) pos(4)];
        set(gca,'Position',posn);
        set(gca,'xticklabel',[]);
    end
    pos = get(gca,'Position');
    
end
export_fig(['frame' sprintf('%02d',ti) '.png'],'-painters');
clf;
end
