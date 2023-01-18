window.onload = function(){
    const h = document.createElement("h1");
    const t = document.createTextNode("Foo");
    h.appendChild(t);
    document.body.appendChild(h);
}

let colorBool = true;
const buttonClickHandler = function() {
    colorBool = !colorBool;
    let e = document.getElementById("btn");
    if (colorBool){
        e.style.color = 'white';
        e.style.backgroundColor = 'blue';
        e.innerHTML = 'FOO';
    }else{
        e.style.color = 'black';
        e.style.backgroundColor = 'Yellow';
        e.innerHTML = 'Bar';
    }
    moveButton();
}

let yPos = -2;

function moveButton() {
    let e = document.getElementById("btn");
    const toto = setInterval(frame, 7);
    function frame() {
        if((!colorBool && yPos === 248) || (colorBool && yPos === -2)){
            clearInterval(toto);
        } else {
            if (!colorBool){
                yPos++;
            }
            else{
                yPos--;
            }
            e.style.top = yPos + 'px';
        }
    }
}

function echoText(str){
    let e = document.getElementById('echo');
    e.innerHTML = str;

}