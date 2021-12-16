import sbt._

object DefinedDependencies {

  private object Versions {
    val akka     = "2.6.16"
    val akkaHttp = "10.2.6"

    val akkaHttpCirce = "1.37.0"
    val akkaHttpCors  = "1.1.2"

    val auth0     = "1.33.0"
    val auth0Jwt  = "3.18.1"
    val auth0Jwks = "0.19.0"

    val aws = "2.17.35"

    val cats = "2.6.1"

    val circe = "0.14.1"

    val commonsCSV  = "1.9.0"
    val commonsLang = "3.12.0"
    val commonsIO   = "2.11.0"

    val doobie = "0.13.4"

    val geotools = "25.2"

    val googleMaps = "1.0.0"

    val hikariCp = "5.0.0"

    val log4j = "2.14.0"

    val postgresql = "42.2.23"

    val pureConfig = "0.16.0"

    val refined = "0.9.27"

    val shapeless = "2.3.7"

  }

  object Akka {
    val actor      = "com.typesafe.akka" %% "akka-actor"       % Versions.akka
    val actorTyped = "com.typesafe.akka" %% "akka-actor-typed" % Versions.akka
    val stream     = "com.typesafe.akka" %% "akka-stream"      % Versions.akka
  }

  object AkkaHttp {
    val http  = "com.typesafe.akka" %% "akka-http"       % Versions.akkaHttp
    val circe = "de.heikoseeberger" %% "akka-http-circe" % Versions.akkaHttpCirce
    val cors  = "ch.megard"         %% "akka-http-cors"  % Versions.akkaHttpCors
  }

  object Apache {
    val commonsCSV  = "org.apache.commons" % "commons-csv"   % Versions.commonsCSV
    val commonsLang = "org.apache.commons" % "commons-lang3" % Versions.commonsLang
    val commonsIO   = "commons-io"         % "commons-io"    % Versions.commonsIO
  }

  object Auth0 {
    val auth0 = "com.auth0" % "auth0"    % Versions.auth0
    val jwt   = "com.auth0" % "java-jwt" % Versions.auth0Jwt
    val jwks  = "com.auth0" % "jwks-rsa" % Versions.auth0Jwks
  }

  object Aws {
    val s3             = "software.amazon.awssdk" % "s3"             % Versions.aws
    val secretsManager = "software.amazon.awssdk" % "secretsmanager" % Versions.aws
  }

  object Cats {
    val core   = "org.typelevel" %% "cats-core"   % Versions.cats
    val kernel = "org.typelevel" %% "cats-kernel" % Versions.cats
  }

  object Circe {
    val core          = "io.circe" %% "circe-core"           % Versions.circe
    val jawn          = "io.circe" %% "circe-jawn"           % Versions.circe
    val parser        = "io.circe" %% "circe-parser"         % Versions.circe
    val literal       = "io.circe" %% "circe-literal"        % Versions.circe
    val numbers       = "io.circe" %% "circe-numbers"        % Versions.circe
    val generic       = "io.circe" %% "circe-generic"        % Versions.circe
    val genericExtras = "io.circe" %% "circe-generic-extras" % Versions.circe
  }

  object Doobie {
    val core     = "org.tpolecat" %% "doobie-core"     % Versions.doobie
    val postgres = "org.tpolecat" %% "doobie-postgres" % Versions.doobie
    val specs2   = "org.tpolecat" %% "doobie-specs2"   % Versions.doobie
    val hikari   = "org.tpolecat" %% "doobie-hikari"   % Versions.doobie
  }

  object GeoTools {
    val gtShapefile = "org.geotools" % "gt-shapefile" % Versions.geotools
  }

  object GoogleMaps {
    val core = "com.google.maps" % "google-maps-services" % Versions.googleMaps
  }

  object HikariCp {
    val hikariCp = "com.zaxxer" % "HikariCP" % Versions.hikariCp
  }

  object Jackson {
    val databind = "com.fasterxml.jackson.core" % "jackson-databind" % "2.12.5"
  }

  object Log4j {
    val api       = "org.apache.logging.log4j" % "log4j-api"          % Versions.log4j
    val core      = "org.apache.logging.log4j" % "log4j-core"         % Versions.log4j
    val slf4jImpl = "org.apache.logging.log4j" % "log4j-slf4j18-impl" % Versions.log4j
  }

  object Postgresql {
    val postgresql = "org.postgresql" % "postgresql" % Versions.postgresql
  }

  object PureConfig {
    val pureConfig = "com.github.pureconfig" %% "pureconfig" % Versions.pureConfig
  }

  object Refined {
    val core          = "eu.timepit" %% "refined"            % Versions.refined
    val cats          = "eu.timepit" %% "refined-cats"       % Versions.refined
    val eval          = "eu.timepit" %% "refined-eval"       % Versions.refined
    val jsonPath      = "eu.timepit" %% "refined-jsonpath"   % Versions.refined
    val pureConfig    = "eu.timepit" %% "refined-pureconfig" % Versions.refined
    val scalaCheck    = "eu.timepit" %% "refined-scalacheck" % Versions.refined
    val refinedScalaz = "eu.timepit" %% "refined-scalaz"     % Versions.refined
    val scodec        = "eu.timepit" %% "refined-scodec"     % Versions.refined
    val refinedScopt  = "eu.timepit" %% "refined-scopt"      % Versions.refined
    val shapeless     = "eu.timepit" %% "refined-shapeless"  % Versions.refined
  }

  object Shapeless {
    val shapeless = "com.chuusai" %% "shapeless" % Versions.shapeless
  }

}
