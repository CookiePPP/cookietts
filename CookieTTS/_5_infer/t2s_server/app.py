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
vocoder_conf  = [[name,details] if os.path.exists(details['modelpath']) else [f"[MISSING]{name}",details] for name, details in list(conf['workers']['MTW']['models'].items())]

# Initialize Flask.
app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')

def process_rq_result(result):
    out = {}
    
    assert result.get('input_text'), "No input_text found in request form!"
    
    text                       =       result.get(    'input_text'           )
    out['speaker']             =       result.getlist('input_speaker'        )# list of speaker(s)
    out['use_arpabet']         = True if result.get(  'input_use_arpabet'    ) == "on" else False
    out['cat_silence_s']       = float(result.get(    'input_cat_silence_s'  ))
    out['batch_size']          =   int(result.get(    'input_batch_size'     ))
    out['max_attempts']        =   int(result.get(    'input_max_attempts'   ))
    out['max_duration_s']      = float(result.get(    'input_max_duration_s' ))
    out['dyna_max_duration_s'] = float(result.get('input_dyna_max_duration_s'))
    out['textseg_len_target']  =   int(result.get('input_textseg_len_target' ))
    out['split_nl']            = True if result.get('input_split_nl'         ) == "on" else False
    out['split_quo']           = True if result.get('input_split_quo'        ) == "on" else False
    out['multispeaker_mode']   =       result.get(  'input_multispeaker_mode')
    
    # (Text) CRLF to LF
    text = text.replace('\r\n','\n')
    
    # (Text) Max Length Limit
    text = text[:int(conf['webpage']['max_input_len'])]
    
    out['text'] = text
    
    return out

@app.route('/tts', methods=['GET', 'POST'])
def texttospeech():
    if request.method == 'POST':
        print("REQUEST RECIEVED")
        # grab all the form inputs
        result = request.form
        
        tts_dict = process_rq_result(result)
        
        MTW_current = result.get('input_MTW_current') # current mel-to-wave
        ttm_current = result.get('input_ttm_current') # current text-to-mel
        print(result)
        
        # update Text-to-mel model if needed
        if t2s.ttm_current != ttm_current:
            t2s.update_tt(ttm_current)
        
        # update Mel-to-Wav model if needed
        #if t2s.MTW_current != MTW_current:
        #    t2s.update_wg(MTW_current)
        
        if False:
            # (Text) Split into segments and send to worker(s)
            pass
            
            # Wait for audio files to be generated or fail
            pass
            
            # Merge the finished product to a single audio file and serve back to user.
            pass
        
        else:
            # generate an audio file from the inputs
            tts_outdict = t2s.infer(**tts_dict)
            print(f"GENERATED {tts_outdict['out_name']}\n\n")
        
        # send updated webpage back to client along with page to the file
        return render_template('main.html',
                                voice=tts_outdict['out_name'],# audio path
                                
                                use_localhost    = conf['webpage']['localhost'],
                                max_input_length = conf['webpage']['max_input_len'],
                                
                                tacotron_conf=tacotron_conf,
                                ttm_current=ttm_current,
                                ttm_len=len(tacotron_conf),
                                
                                vocoder_conf=vocoder_conf,
                                MTW_current=MTW_current,
                                MTW_len=len(vocoder_conf),
                                
                                sp_len=len(speakers),
                                speakers_available_short=[sp.split("_")[-1] for sp in speakers],
                                speakers_available=speakers,
                                
                                current_text  =tts_dict['text'],
                                sample_text   =conf['webpage']['defaults']['background_text'],
                                speaker=tts_dict['speaker'][0],
                                
                                use_arpabet         = tts_dict['use_arpabet'],
                                cat_silence_s       = tts_dict['cat_silence_s'],
                                batch_size          = tts_dict['batch_size'],
                                max_attempts        = tts_dict['max_attempts'],
                                max_duration_s      = tts_dict['max_duration_s'],
                                dyna_max_duration_s = tts_dict['dyna_max_duration_s'],
                                textseg_len_target  = tts_dict['textseg_len_target'],
                                split_nl            = tts_dict['split_nl'],
                                split_quo           = tts_dict['split_quo'],
                                multispeaker_mode   = tts_dict['multispeaker_mode'],
                                
                                gen_time    = f'{tts_outdict["time_to_gen"]:.1f}',
                                gen_dur     = f'{tts_outdict["audio_seconds_generated"]:.1f}',
                                total_specs = f'{tts_outdict["total_specs"]:.0f}',
                                n_passes    = f'{tts_outdict["n_passes"]:.0f}',
                                avg_score   = f'{tts_outdict["avg_score"]:.3f}',
                                rtf         = f'{tts_outdict["rtf"]:.2f}',
                                fail_rate   = f'{tts_outdict["fail_rate"]*100.:.1f}',
                              )

#Route to render GUI
@app.route('/')
def show_entries():
    return render_template('main.html',
                            voice=None,
                            
                            use_localhost    = conf['webpage']['localhost'],
                            max_input_length = conf['webpage']['max_input_len'],
                            
                            tacotron_conf=tacotron_conf,
                            ttm_current=conf['workers']['TTM']['default_model'],
                            ttm_len=len(tacotron_conf),
                            
                            vocoder_conf=vocoder_conf,
                            MTW_current=conf['workers']['MTW']['default_model'],
                            MTW_len=len(vocoder_conf),
                            
                            sp_len=len(speakers),
                            speakers_available_short=[sp.split("_")[-1] for sp in speakers],
                            speakers_available=speakers,
                            
                            current_text        = conf['webpage']['defaults']['current_text'],
                            sample_text         = conf['webpage']['defaults']['background_text'],
                            speaker             = conf['webpage']['defaults']['speaker'],
                            use_arpabet         = conf['webpage']['defaults']['use_arpabet'],
                            cat_silence_s       = conf['webpage']['defaults']['cat_silence_s'],
                            batch_size          = conf['webpage']['defaults']['batch_size'],
                            max_attempts        = conf['webpage']['defaults']['max_attempts'],
                            max_duration_s      = conf['webpage']['defaults']['max_duration_s'],
                            dyna_max_duration_s = conf['webpage']['defaults']['dyna_max_duration_s'],
                            textseg_len_target  = conf['webpage']['defaults']['textseg_len_target'],
                            split_nl            = conf['webpage']['defaults']['split_nl'],
                            split_quo           = conf['webpage']['defaults']['split_quo'],
                            multispeaker_mode   = conf['webpage']['defaults']['multispeaker_mode'],
                            gen_time   = "",
                            gen_dur    = "",
                            total_specs= "",
                            n_passes   = "",
                            avg_score  = "",
                            fail_rate  = "",)

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
