build_path=../../graphmodels-build

$build_path/bin/test_reinforce
$build_path/bin/test_text_gen ../graphmodels/data/test.txt rnn ../graphmodels/test/out_rnn.txt
$build_path/bin/test_text_gen ../graphmodels/data/test.txt lstm ../graphmodels/test/out_lstm.txt

