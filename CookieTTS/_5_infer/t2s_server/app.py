import sys
from flask import Flask,render_template, Response, request, send_from_directory, url_for
# Tornado web server
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
import tornado.ioloop
from tornado.ioloop import IOLoop
from text2speech import T2S
import os
import json

# load T2S config
with open('t2s_config.json', 'r') as f:
    conf = json.load(f)

# start worker(s)
t2s = T2S(conf['workers'])
speakers = [x for x in list(t2s.ttm_sp_name_lookup.keys()) if "(Music)" not in x]
if conf['webpage']['sort_speakers']:
    speakers = sorted(speakers)
tacotron_conf = [[name,details] if os.path.exists(details['modelpath']) else [f"[MISSING]{name}",details] for name, details in list(conf['workers']['TTM']['models'].items())]
waveglow_conf = [[name,details] if os.path.exists(details['modelpath']) else [f"[MISSING]{name}",details] for name, details in list(conf['workers']['MTW']['models'].items())]

# Initialize Flask.
app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')

@app.route('/tts', methods=['GET', 'POST'])
def texttospeech():
    if request.method == 'POST':
        print("REQUEST RECIEVED")
        # grab all the form inputs
        result = request.form
        
        assert result.get('input_text'), "No input_text found in request form!"
        
        speaker = result.getlist('input_speaker')
        text = result.get('input_text')
        style_mode = result.get('input_style_mode')
        textseg_mode = result.get('input_textseg_mode')
        batch_mode = result.get('input_batch_mode')
        max_attempts = int(result.get('input_max_attempts')) if result.get('input_max_attempts') else 256
        max_duration_s = float(result.get('input_max_duration_s'))
        batch_size = int(result.get('input_batch_size'))
        dyna_max_duration_s = float(result.get('input_dyna_max_duration_s'))
        use_arpabet = True if result.get('input_use_arpabet') == "on" else False
        target_score = float(result.get('input_target_score'))
        multispeaker_mode = result.get('input_multispeaker_mode')
        cat_silence_s = float(result.get('input_cat_silence_s'))
        textseg_len_target = int(result.get('input_textseg_len_target'))
        MTW_current = result.get('input_MTW_current') # current mel-to-wave
        ttm_current = result.get('input_ttm_current') # current text-to-mel
        print(result)
        
        # update Text-to-mel model if needed
        if t2s.ttm_current != ttm_current:
            t2s.update_tt(ttm_current)
        
        # update Mel-to-Wav model if needed
        #if t2s.MTW_current != MTW_current:
        #    t2s.update_wg(MTW_current)
        
        # (Text) CRLF to LF
        text = text.replace('\r\n','\n')
        
        # (Text) Max Length Limit
        text = text[:int(conf['webpage']['max_input_len'])]
        
        # (Text) Split into segments and send to worker(s)
        pass
        
        # Wait for audio files to be generated or fail
        pass
        
        # Merge the finished product to a single audio file and serve back to user.
        pass
        
        # generate an audio file from the inputs
        filename, gen_time, gen_dur, total_specs, n_passes, avg_score = t2s.infer(text, speaker, style_mode, textseg_mode, batch_mode, max_attempts, max_duration_s, batch_size, dyna_max_duration_s, use_arpabet, target_score, multispeaker_mode, cat_silence_s, textseg_len_target)
        print(f"GENERATED {filename}\n\n")
        
        # send updated webpage back to client along with page to the file
        return render_template('main.html',
                                use_localhost=conf['webpage']['localhost'],
                                max_input_length=conf['webpage']['max_input_len'],
                                tacotron_conf=tacotron_conf,
                                ttm_current=ttm_current,
                                ttm_len=len(tacotron_conf),
                                waveglow_conf=waveglow_conf,
                                MTW_current=MTW_current,
                                MTW_len=len(waveglow_conf),
                                sp_len=len(speakers),
                                speakers_available_short=[sp.split("_")[-1] for sp in speakers],
                                speakers_available=speakers,
                                current_text=text,
                                voice=filename,
                                sample_text=conf['webpage']['defaults']['background_text'],
                                speaker=speaker,
                                style_mode=style_mode,
                                textseg_mode=textseg_mode,
                                batch_mode=batch_mode,
                                max_attempts=max_attempts,
                                max_duration_s=max_duration_s,
                                batch_size=batch_size,
                                dyna_max_duration_s=dyna_max_duration_s,
                                use_arpabet=result.get('input_use_arpabet'),
                                target_score=target_score,
                                gen_time=round(gen_time,2),
                                gen_dur=round(gen_dur,2),
                                total_specs=total_specs,
                                n_passes=n_passes,
                                avg_score=round(avg_score,3),
                                multispeaker_mode=multispeaker_mode,
                                cat_silence_s=cat_silence_s,
                                textseg_len_target=textseg_len_target,)

#Route to render GUI
@app.route('/')
def show_entries():
    return render_template('main.html',
                            use_localhost=conf['webpage']['localhost'],
                            max_input_length=conf['webpage']['max_input_len'],
                            tacotron_conf=tacotron_conf,
                            ttm_current=conf['workers']['TTM']['default_model'],
                            ttm_len=len(tacotron_conf),
                            waveglow_conf=waveglow_conf,
                            MTW_current=conf['workers']['MTW']['default_model'],
                            MTW_len=len(waveglow_conf),
                            sp_len=len(speakers),
                            speakers_available_short=[sp.split("_")[-1] for sp in speakers],
                            speakers_available=speakers,
                            current_text=conf['webpage']['defaults']['current_text'],
                            sample_text=conf['webpage']['defaults']['background_text'],
                            voice=None,
                            speaker=conf['webpage']['defaults']['speaker'],
                            style_mode=conf['webpage']['defaults']['style_mode'],
                            textseg_mode=conf['webpage']['defaults']['textseg_mode'],
                            batch_mode=conf['webpage']['defaults']['batch_mode'],
                            max_attempts=conf['webpage']['defaults']['max_attempts'],
                            max_duration_s=conf['webpage']['defaults']['max_duration_s'],
                            batch_size=conf['webpage']['defaults']['batch_size'],
                            dyna_max_duration_s=conf['webpage']['defaults']['dyna_max_duration_s'],
                            use_arpabet=conf['webpage']['defaults']['use_arpabet'],
                            target_score=conf['webpage']['defaults']['target_score'],
                            gen_time="",
                            gen_dur="",
                            total_specs="",
                            n_passes="",
                            avg_score="",
                            multispeaker_mode=conf['webpage']['defaults']['multispeaker_mode'],
                            cat_silence_s=conf['webpage']['defaults']['cat_silence_s'],
                            textseg_len_target=conf['webpage']['defaults']['textseg_len_target'],)

#Route to stream audio
@app.route('/<voice>', methods=['GET'])
def streammp3(voice):
    print("AUDIO_REQUEST: ", request)
    def generate():
        with open(os.path.join(t2s.conf['output_directory'], voice), "rb") as fwav:# open audio_path
            data = fwav.read(1024)
            while data:
                yield data
                data = fwav.read(1024)
    
    stream_audio = False
    if stream_audio:# don't have seeking working atm
        return Response(generate(), mimetype="audio/wav")
    else:
        return send_from_directory(t2s.conf['output_directory'], voice)


#launch a Tornado server with HTTPServer.
if __name__ == "__main__":
    port = 5000
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(port)
    io_loop = tornado.ioloop.IOLoop.current()
    io_loop.start()