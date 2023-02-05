const input_dir = "/etc/sherlockpipe/input/";
const working_dir = "/etc/sherlockpipe/working/";
const output_dir = "/etc/sherlockpipe/output/";
const refresh_button = document.getElementById("refresh");
const statusContainer = document.getElementById("statusContainer");
const inputContainer = document.getElementById("inputContainer");
const workingContainer = document.getElementById("workingContainer");
const outputContainer = document.getElementById("outputContainer");

function listDir() {
    //cockpit.spawn(["journalctl", "-u", "sherlock-aas", "|", "tail", "-n", "25"])
    //    .stream(statusSuccess)
    //    .then(statusSuccess)
    //    .catch(statusFail);    
    statusContainer.innerHTML = "";
    inputContainer.innerHTML = "";
    workingContainer.innerHTML = "";
    outputContainer.innerHTML = "";
    cockpit.spawn(["tree", "-fid", "-L", "2", input_dir])
        .stream(inputSuccess)
    //    .then(inputSuccess)
        .catch(inputFail);
    cockpit.spawn(["tree", "-fid", "-L", "2", working_dir])
        .stream(workingSuccess)
    //    .then(workingSuccess)
        .catch(workingFail);
    cockpit.spawn(["tree", "-fid", "-L", "2", output_dir])
        .stream(outputSuccess)
    //    .then(outputSuccess)
        .catch(outputFail);
    cockpit.spawn(["journalctl", "-u", "sherlock-aas", "-n", "10"], {"superuser": "try"})
        .stream(statusSuccess)
    //    .then(statusSuccess)
        .catch(statusFail);
}

refresh_button.addEventListener("click", listDir);
//refresh_button.addEventListener("load", listDir);

function statusSuccess(data) {
    statusContainer.append(document.createTextNode(data));
}

function inputSuccess(data) {
    inputContainer.append(document.createTextNode(data));
}

function workingSuccess(data) {
    workingContainer.append(document.createTextNode(data));
}

function outputSuccess(data) {
    outputContainer.append(document.createTextNode(data));
}

function statusFail() {
    statusContainer.innerHTML = "fail";
}

function inputFail() {
    inputContainer.innerHTML = "fail";
}

function workingFail() {
    workingContainer.innerHTML = "fail";
}

function outputFail() {
    outputContainer.innerHTML = "fail";
}

listDir();

// Send a 'init' message.  This tells integration tests that we are ready to go
cockpit.transport.wait(function() { });
