package org.nimbleedge.recoedge

import models._

import akka.actor.typed.ActorRef
import akka.actor.typed.Behavior
import akka.actor.typed.scaladsl.Behaviors
import akka.actor.typed.scaladsl.ActorContext
import akka.actor.typed.scaladsl.AbstractBehavior
import akka.actor.typed.Signal
import akka.actor.typed.PostStop

object Aggregator {
    def apply(aggId: AggregatorIdentifier): Behavior[Command] =
        Behaviors.setup(new Aggregator(_, aggId))
    
    trait Command

    // In cae of any Trainer / Aggregator (Chile) Termination
    private final case class AggregatorTerminated(actor: ActorRef[Aggregator.Command], aggId: AggregatorIdentifier)
        extends Aggregator.Command
    
    private final case class TrainerTerminated(actor: ActorRef[Trainer.Command], traId: TrainerIdentifier)
        extends Aggregator.Command

    // TODO
    // Add messages here
}

class Aggregator(context: ActorContext[Aggregator.Command], aggId: AggregatorIdentifier) extends AbstractBehavior[Aggregator.Command](context) {
    import Aggregator._
    import Supervisor.{ RequestTrainer, RequestTopology }

    // TODO
    // Add state and persistent information

    context.log.info("Aggregator {} started", aggId.toString())

    override def onMessage(msg: Command): Behavior[Command] =
        msg match {
            case trackMsg @ RequestTrainer(requestId, traId, replyTo) =>
                // TODO
                this

            case trackMsg @ RequestTopology(requestId, entity, replyTo) =>
                // TODO
                this
            
            case AggregatorTerminated(actor, aggId) =>
                context.log.info("Aggregator with id {} has been terminated", aggId.toString())
                // TODO
                this
            
            case TrainerTerminated(actor, traId) =>
                context.log.info("Trainer with id {} has been terminated", traId.toString())
                // TODO
                this
        }
    
    override def onSignal: PartialFunction[Signal,Behavior[Command]] = {
        case PostStop =>
            context.log.info("Aggregator {} stopped", aggId.toString())
            this
    }
}