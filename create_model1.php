<?php
ini_set('error_reporting', E_ALL);
ini_set('display_errors', 1);
require_once __DIR__ . '/vendor/autoload.php';

use Phpml\Dataset\ArrayDataset;
use Phpml\Classification\NaiveBayes;
use Phpml\Classification\KNearestNeighbors;
use Phpml\FeatureExtraction\TokenCountVectorizer;
use Phpml\Tokenization\WhitespaceTokenizer;
use Phpml\Tokenization\WordTokenizer;
use Phpml\FeatureExtraction\TfIdfTransformer;
use Phpml\FeatureExtraction\StopWords\Thai;
use Phpml\Tokenization\NGramTokenizer;
// $samples = [[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]];
$samples = array();
$text = array(
  array(
    'ทดสอบฝึกพื้นฐานคอมพิวเตอร์','ทดสอบฝึกพื้นฐานภาษาอังกฤษ','ฝึกทดสอบ','ทดสอบพื้นฐาน','การเข้าทดสอบ',
    'การเข้าทดสอบก่อนจบ','ทดสอบคอมพิวเตอร์','การสอบคอมพิวเตอร์','การสอบภาษาอังกฤษ','สอบคอมพิวเตอร์','สอบภาษาอังกฤษ'
  ),
  array(
    'การเข้าใช้งานอินเทอร์เน็ต','การใช้งานไวไฟ','เข้าใช้อินเทอร์เน็ต','การใช้งานเน็ต'
    // ,'การใช้ VPN','การใช้งาน Eduroam','Eduroam','VPN'
  ),
  array(
    'พัฒนาเว็บไซต์','พัฒนาระบบสารสนเทศและเทคโนโลยี','แก้ไขเว็บไซต์','แก้ไขข่าว','วิเคราะห์ระบบ','วิเคราะห์งาน','แก้ไขระบบผลงานวิชาการ','ระบบผลงานวิชาการ'
  ),
  array(
    'ซ่อมบำรุง ปรับปรุง ระบบ คอมพิวเตอร์','ปรับปรุง','ซ่อมคอมพิวเตอร์','ซ่อมเครื่องพิมพ์','ติดตั้งซอฟต์แวร์','ตั้งคอมพิวเตอร์','ติดตั้งระบบเครือข่าย','ติดตั้งไวไฟ','ติดตั้งระบบโทรศัพท์'
  ),
);
$labels = array('drudigital', 'internet', 'systems', 'maintainance');

// foreach ($text as $key => $value) {
//   // echo $value[0]."<br/>";
//   array_push($samples,$value[0].' ,'.$value[1]);
//   // array_push($samples,array($value[0],$value[1]));
// }
$samples = $text;
// echo "<pre>";
// var_dump($samples);
// var_dump($labels);
// exit;
$dataset = new ArrayDataset($samples,$labels);

var_dump($dataset);

$tokenize = new WordTokenizer();

// $vectorizer = new TokenCountVectorizer($tokenize,new Thai());
$vectorizer = new TokenCountVectorizer(new NGramTokenizer(1, 200),new Thai());
// var_dump($vectorizer);
// exit;
$vectorizer->fit($dataset->getSamples(),$dataset->getTargets());
// $getdata = $vectorizer->fit($dataset->getSamples(),$dataset->getTargets());
// var_dump($getdata);
// exit;
$vocabulary = $vectorizer->getVocabulary();
var_dump($vocabulary);
?>
