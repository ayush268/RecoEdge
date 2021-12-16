name := "RecoEdge"

version := "0.1"

scalaVersion := "2.13.6"

projectDependencies ++= Seq(
//  DefinedDependencies.PureConfig.pureConfig,
//  //------------------------
//  DefinedDependencies.Cats.core,
//  DefinedDependencies.Cats.kernel,
//  //------------------------
//  DefinedDependencies.Shapeless.shapeless,
//  //------------------------
//  DefinedDependencies.Refined.core,
//  DefinedDependencies.Refined.cats,
//  DefinedDependencies.Refined.eval,
//  //------------------------
//  DefinedDependencies.Circe.core,
//  DefinedDependencies.Circe.jawn,
//  DefinedDependencies.Circe.parser,
//  DefinedDependencies.Circe.literal,
//  DefinedDependencies.Circe.numbers,
//  DefinedDependencies.Circe.generic,
//  DefinedDependencies.Circe.genericExtras,
  //------------------------
  DefinedDependencies.Akka.actor,
  DefinedDependencies.Akka.actorTyped,
  //------------------------
  DefinedDependencies.AkkaHttp.http,
  DefinedDependencies.AkkaHttp.circe,
  DefinedDependencies.AkkaHttp.cors
//  //------------------------
//  DefinedDependencies.GeoTools.gtShapefile,
//  //------------------------
//  DefinedDependencies.GoogleMaps.core,
//  //------------------------
//  DefinedDependencies.Apache.commonsLang,
//  DefinedDependencies.Apache.commonsIO,
//  DefinedDependencies.Apache.commonsCSV,
//  //------------------------
//  DefinedDependencies.Postgresql.postgresql,
//  //------------------------
//  DefinedDependencies.HikariCp.hikariCp,
//  //------------------------
//  DefinedDependencies.Aws.s3,
//  //------------------------
//  DefinedDependencies.Log4j.api,
//  DefinedDependencies.Log4j.core,
//  DefinedDependencies.Log4j.slf4jImpl,
//  //------------------------
//  DefinedDependencies.Jackson.databind
)
