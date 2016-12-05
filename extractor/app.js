var express = require('express');
var path = require('path');
var favicon = require('serve-favicon');
var logger = require('morgan');
var cookieParser = require('cookie-parser');
var bodyParser = require('body-parser');
var mongoose = require('mongoose');

mongoose.set('debug', true);

var index = require('./routes/index');
var users = require('./routes/users');
var http = require('http');

var app = express();
var inflight = 0;
var connectionString = 'mongodb://127.0.0.1:27017/wam-fall-2016';
mongoose.connect(connectionString);
app.use(logger('dev'));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/', index);
app.use('/users', users);

// catch 404 and forward to error handler
app.use(function(req, res, next) {
  var err = new Error('Not Found');
  err.status = 404;
  next(err);
});

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

module.exports = app;

var userSchema = new mongoose.Schema({
	userName : String,
	userId: {type: String, unique: true, required: true},
	matches: [String]
});


var MatchModel = mongoose.model('MatchModel',new mongoose.Schema({
        matchId: String,
        teamOneHeroes: [{type: String}],
        teamTwoHeroes: [{type: String}],
        winner: Number
    }
));

var APIKEY = 'KXTJ3EKVUBQAF61O';
var matchCount = 0;
var matchList = [];
var matchDetailsList = [];

function sleep(milliSeconds){
    var waitUntil = new Date().getTime() + milliSeconds;
    while(new Date().getTime() < waitUntil) true;
}

function getMatchDetails(persistedUser) {
    var url = "http://api.heroesofnewerth.com/match_history/ranked/accountid/" + persistedUser.userId + "/?token=" + APIKEY;
    sleep(5000);
    http.get(url, (response) => {
        var result = '';
        response.setEncoding('utf8');
        response.on('data', function(chunk) {
            result = result + chunk;
        });
        response.on('end', function () {
            try {
                var users = JSON.parse(result);
            }catch(e){
                console.log("Error occured while parsing 1: " + result); //error in the above string(in this case,yes)!
                return;
            }
            historyList = users[0]['history'].split(',');
            historyList.forEach((history) => {
                matchCount = matchCount + 1;
                var matchId = history.split('|')[0];
                matchList.push(matchId);
                getMatchDetailsHelper(matchId);
            });

            var newDocument = persistedUser;
            newDocument.matches = matchList;
            console.log("++++++++++++++++++++++++++++");
            console.log("Match Count = " + matchCount);
            console.log("Match List = " + matchList.toString());
            console.log("++++++++++++++++++++++++++++");
        });
    });
}

function getMatchDetailsHelper (matchId) {
    var url = "http://api.heroesofnewerth.com/match/statistics/matchid/" + matchId.toString() + "/?token=" + APIKEY;
    console.log("URL = " + url)
    inflight = inflight + 1;
    http.request(url, (response) => {
        var result = '';
        response.setEncoding('utf8');
        response.on('data', function (chunk) {
            result = result + chunk;
        });
        response.on('end', function () {
            try {
                var matchDetails = JSON.parse(result);
            } catch (e) {
                if (result == "Too many requests") {
                    sleep(1000);
                    console.log('Handling too many requests');
                    console.log(url);
                    return getMatchDetailsHelper (matchId);
                }
                console.log("Error occured while parsing 2:" + result); //error in the above string(in this case,yes)!
                console.log(url);
                run();
                return;
            }
            var processedMatchDetails = {
                matchId: '',
                teamOneHeroes: [],
                teamTwoHeroes: [],
                winner: 0
            }
            matchDetails.forEach((matchDetail) => {
                processedMatchDetails['matchId'] = matchDetail.match_id;
                if (matchDetail.team == "1") {
                    processedMatchDetails['teamOneHeroes'].push(matchDetail.hero_id);
                    if (matchDetail.wins == "1") {
                        processedMatchDetails['winner'] = 1;
                    }
                } else if (matchDetail.team == "2") {
                    processedMatchDetails['teamTwoHeroes'].push(matchDetail.hero_id);
                    if (matchDetail.wins == "1") {
                        processedMatchDetails['winner'] = 2;
                    }
                }
            });
            matchDetailsList.push(processedMatchDetails);
            console.log(processedMatchDetails);
            MatchModel.create(processedMatchDetails, function(error, data) {
                if (error) {
                    console.log("Error Occured where persisting Data");
                    console.log(data);
                    console.log(error);
                    run();
                    return;
                }
                run();
            });
            console.log("++++++++++++++++++++++++++++");
            console.log("processedMatchDetails = " + processedMatchDetails.toString());
            console.log(processedMatchDetails);
            console.log("Match Details List = " + matchDetailsList.toString());
            console.log(matchDetailsList);
            inflight = inflight - 1;
            console.log("++++++++++++++++++++++++++++");
        });
    }).end();
}


function uploadUserDetails() {
	var userNames = ['Fly', 'Nova', 'Era', 'Protail', 'Notail', 'Crowtail', 'BKeeeeD', 'Freshpro','Moonmeander', 'Testie', 'H4nn1', 'dososolah', 'bkid', 'chu`', 'angrytestie', 'moonmeander', 'Jiggle_billy', 'swindlemelonzz', 'KheZu', 'Zlapped', 'm`ICKe', 'fUzi', 'Handsken', 'BeaverBanger', 'Flensmeister', 'Limmp', 'Mynuts', 'Jonassomfan', 'BOXl', 'Zfreek', 'Sealkid'];
    userNames.forEach((user) => {
        sleep(1000);
        var apiString = 'http://api.heroesofnewerth.com/player_statistics/all/nickname/' + user + '/?token=' + APIKEY;
        http.get(apiString, (response) => {
            var result = '';
            response.setEncoding('utf8');
            response.on('data', function(chunk) {
                result = result + chunk;
            });
            response.on('end', function () {
                var temp = JSON.parse(result);
                var account_id = temp["account_id"];
                document = {userName: user, userId: account_id};
                getMatchDetails(document);
            });
        });
    });
}

// uploadUserDetails();
var runNumber=147104034;
function run() {
    try {
        runNumber = runNumber + 1;
        if (runNumber < 147400010) {
            getMatchDetailsHelper(runNumber);
        }
    } catch(err) {
        run()
    }
}

run();

