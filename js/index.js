const onmainbuttonclick = async (event) => {

    // Main video area
    var videoinput = document.getElementById('main-video-area');

    // Setting page size
    videoinput.width = 800;
    videoinput.height = 600;

    // Open web camera
    var stream = await navigator.mediaDevices.getUserMedia({ video: { width: 800, height: 600 }, audio: true });
    videoinput.srcObject = stream;

    // Play video from webcam
    videoinput.play();
};

const drawcanvas_videoframe = () => {
    var videoinput = document.getElementById('main-video-area');
    var canvas = document.getElementById('main-canvas-area');
    var ctx = canvas.getContext('2d');

    if(videoinput.paused || videoinput.ended) return false;

    canvas.width = videoinput.width + 10;
    canvas.height = videoinput.height + 10;
    canvas.style.width = videoinput.width + 10;
    canvas.style.height = videoinput.height + 10;

    // Draw Video
    ctx.drawImage(videoinput, 0, 0, canvas.width, canvas.height);
    
    requestAnimationFrame(drawcanvas_videoframe);
};

const onvideotagplaying = (event) => {
    requestAnimationFrame(drawcanvas_videoframe);
};

const onstopbuttonclick = (event) => {
    var videoinput = document.getElementById('main-video-area');
    var stop_button = document.getElementById('stop-video');
 
    if (!videoinput.paused) {
        videoinput.pause();
        stop_button.innerText = 'Play Camera';
    } else {
        videoinput.play();
        stop_button.innerText = 'Pause Camera';
    }
};

const oncapturefaceclick = (event) => {
    event.preventDefault();
    detect_face('main-canvas-area', 1, 32);
};

/*var main_button = document.getElementById('open-video');
main_button.addEventListener('click', onmainbuttonclick);

var stop_button = document.getElementById('stop-video');
stop_button.addEventListener('click', onstopbuttonclick);*/

function make_base()
{
    var canvas = document.getElementById('main-canvas-area'),
    context = canvas.getContext('2d');
    base_image = new Image(800, 450);
    canvas.width = 800;
    canvas.height = 450;
    base_image.src = 'up_saved.jpg';
    base_image.onload = function()  {
        context.drawImage(base_image, 0, 0);
    }
}

var capture_face = document.getElementById('capture-face');
capture_face.addEventListener('click', oncapturefaceclick);

make_base();
// Video Event
//var videoinput = document.getElementById('main-video-area');
//videoinput.addEventListener('playing', onvideotagplaying);
