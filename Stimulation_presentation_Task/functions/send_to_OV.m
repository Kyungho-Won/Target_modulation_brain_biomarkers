function send_to_OV(fid, stim_identifer)
% send stim to OpenViBE
% OpenViBE software tagging
% OV 2.0.1 and later tag format to send to the server
% openvibeExternalStimulations
% TCP_Tagging_Port 15361
% [uint64 flags ; uint64 stimulation_identifier ; uint64 timestamp]

OV_flags = 4;
val64as32 = swapbytes(typecast(uint64(OV_flags), 'uint32'));
% Matlab fwrite does not support uint64, then split uint64 into uint32s
fwrite(fid, (val64as32), 'uint32');

OV_stim_identifier = stim_identifer;
val64as32 = swapbytes(typecast(uint64(OV_stim_identifier), 'uint32'));
fwrite(fid, (val64as32), 'uint32');

OV_time_stamp = 0;
val64as32 = swapbytes(typecast(uint64(OV_time_stamp), 'uint32'));
fwrite(fid, (val64as32), 'uint32');

end