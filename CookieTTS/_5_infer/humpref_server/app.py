import sys
from CookieTTS.utils import get_args, force
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
with open('default_config.json', 'r') as f:
    conf = json.load(f)

# start worker(s)
t2s = T2S(conf['workers'])
vocoder_conf  = [[name,details] if os.path.exists(details['modelpath']) else [f"[MISSING]{name}",details] for name, details in list(conf['workers']['MTW']['models'].items())]

# Initialize Flask.
app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')

def process_rq_result(result):
    out = {}
    
    out['samples_to_compare'] =     int(result.get('input_samples_to_compare'))
    out['same_speaker'      ] = True if result.get('input_same_speaker'      ) == "on" else False
    out['same_transcript'   ] = True if result.get('input_same_transcript'   ) == "on" else False
    
    out['whitelist_speakers'] = result.getlist('input_speaker')
    
    out['audiopaths'] = result.get('input_audiopaths', None)
    if out['audiopaths'] is not None:
        out['audiopaths'] = json.loads(out['audiopaths'])
    
    out['input_best_audio'] = result.get('input_best_audio',   None)
    if out['input_best_audio'] is not None:
        # "Audio #1 is better" -> 0
        # "Audio #2 is better" -> 1
        out['input_best_audio'] = int(out['input_best_audio'].split("Audio #")[1].split(" is better")[0])-1
    return out

@app.route('/tts', methods=['GET', 'POST'])
def texttospeech():
    # if no information sent, give the user the homepage
    if request.method != 'POST':
        return show_entries()
    
    # grab all the form inputs
    result = request.form
    print("#"*79+f'\n{result}\n'+"#"*79)
    
    tts_dict = process_rq_result(result)
    
    # if an audio file was rated last iter.
    if tts_dict['input_best_audio'] is not None and tts_dict['audiopaths'] is not None:
        t2s.write_best_audio(tts_dict['audiopaths'], tts_dict['input_best_audio'])
    
    # update Mel-to-Wav model if needed
    MTW_current = result.get('input_MTW_current') # current mel-to-wave
    if t2s.MTW_current != MTW_current:
        t2s.update_hifigan(MTW_current)
    
    # get spectrograms + speaker name + reconstruct any GT audio.
    tts_outdict = t2s.get_samples(tts_dict)
    
    # send updated webpage back to client along with page to the file
    return render_template(
        'main.html',
        audiopaths_str = json.dumps(tts_outdict['absaudiopaths']),
        audiopaths  = tts_outdict['audiopaths'],
        spectpaths  = tts_outdict['spectpaths'],
        speakers    = tts_outdict['speakers'],
        transcripts = tts_outdict['transcripts'],
        
        speakers_selected = tts_dict['whitelist_speakers'],
        speakers_available       = tts_outdict['possible_speakers'],
        speakers_available_short = tts_outdict['possible_speakers'],
        
        vocoder_conf=vocoder_conf,
        MTW_current=MTW_current,
        MTW_len=len(vocoder_conf),
        
        samples_to_compare = tts_dict['samples_to_compare'],
        same_speaker       = tts_dict['same_speaker'],
        same_transcript    = tts_dict['same_transcript'],)

#Route to render GUI
@app.route('/')
def show_entries():
    return render_template(
        'main.html',
        audiopaths = [],
        spectpaths = [],
        speakers   = [],
        transcripts= [],
        
        vocoder_conf=vocoder_conf,
        MTW_current=conf['workers']['MTW']['default_model'],
        MTW_len=len(vocoder_conf),
        
        samples_to_compare = conf['webpage']['defaults']['samples_to_compare'],
        same_speaker       = conf['webpage']['defaults']['same_speaker'],
        same_transcript    = conf['webpage']['defaults']['same_transcript'],)

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
    port = 5002
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(port)
    io_loop = tornado.ioloop.IOLoop.current()
    io_loop.start()
