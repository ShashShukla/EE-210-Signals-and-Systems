clear; close; clc;

//input wav file
[inp, fs_inp] = wavread("BheegiRegular-part.wav");

//loading 2 channel RIR
rir = wavread("long_echo_hall_16k.wav");

//SINGLE CHANNEL CONVOLUTION FUNCTION
function[output] = fun_conv(inp1,rir1)
    //PASTE YOUR CODE FOR CONVOLUTION OF TWO SIGNALS
    output = convol(inp1, rir1)
    //output = conv(inp1,rir1);
endfunction


//obtain RIR for left channel
rir_left = rir(1,:);
//obtain RIR for right channel
rir_right = rir(2,:);

//obtain convolved signal for left channel
out_left = fun_conv(inp,rir_left);
//obtain convolved signal for right channel
out_right = fun_conv(inp,rir_right);

//obtaining stereo sound by combining two channels
out = [out_left;out_right];
out = out/max(abs(out));

//playing convolved signal
playsnd(out,fs_inp);

//writing convolved signal
wavwrite(out,fs_inp,"echo_hall_convolved")
